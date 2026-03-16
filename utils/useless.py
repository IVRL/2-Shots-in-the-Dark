def remove_banding_directional(dark_frame, method='mean', row_window=5, gaussian_sigma=30):
    """
    Enhanced banding removal with options for different estimators.
    
    Parameters:
    dark_frame (np.array): Dark frame with shape (4, H, W)
    method (str): 'mean', 'median', or 'filtered'
    row_window (int): Window size for row filtering (for 'filtered' method)
    
    Returns:
    np.array: Processed dark frame with shape (4, H, W)
    """
    dark_frame_blurred = apply_gaussian_blur(dark_frame, sigma=gaussian_sigma)
    processed = dark_frame - dark_frame_blurred
    processed = dark_frame.copy()
    
    for c in range(dark_frame.shape[0]):
        channel_data = dark_frame[c]
        h, w = channel_data.shape
        
        if method == 'mean':
            # Simple row mean subtraction
            row_means = np.mean(channel_data, axis=1, keepdims=True)
            processed[c] = channel_data - row_means
            
        elif method == 'median':
            # Use median for more robust estimation 
            row_medians = np.median(channel_data, axis=1, keepdims=True)
            processed[c] = channel_data - row_medians
            
        elif method == 'filtered':
            # Apply smoothing to the row means to handle noise
            row_means = np.mean(channel_data, axis=1)
            
            # Smooth the row means with a moving average
            smoothed_means = np.convolve(
                row_means, 
                np.ones(row_window)/row_window, 
                mode='same'
            )
            
            # Subtract the smoothed row means
            processed[c] = channel_data - smoothed_means[:, np.newaxis]

    processed = processed + dark_frame_blurred
    
    return processed


def extract_darkshading(reference, method='mean', row_window=5, sigma=30):
    reference_without_banding = remove_banding_directional(reference, method=method, row_window=row_window, gaussian_sigma=sigma)
    darkshading = apply_gaussian_blur(reference_without_banding, sigma=sigma)
    return darkshading




# --------------------------------------------
# denoising
# --------------------------------------------
sys.path.append(os.getcwd())
from DPIR.models.network_unet import UNetRes as net


def estimate_sigma(channel):
    """
    Estimate noise standard deviation using the median absolute deviation method.
    This is a robust method for estimating Gaussian noise sigma.
    
    Args:
        channel (numpy.ndarray): A 2D array representing a single image channel (normalized 0-1)
        
    Returns:
        float: Estimated noise standard deviation (in the 0-1 range)
    """
    # Calculate the median of the channel
    median = np.median(channel)
    
    # Calculate the median absolute deviation (MAD)
    mad = np.median(np.abs(channel - median))
    
    # Estimate sigma using the relationship: sigma = MAD / 0.6745
    # This conversion factor assumes Gaussian noise
    sigma_est = mad / 0.6745
    
    return max(sigma_est, 0.01)  # Ensure a minimum value to avoid issues

def denoise_4channel_image(image_4channel, model, device):
    """
    Denoise a 4-channel image by processing each channel separately with DRUNet.
    
    Args:
        image_4channel (numpy.ndarray): Input image with shape (4, H, W), normalized to [0, 1]
        model_path (str): Path to the DRUNet grayscale model weights
        
    Returns:
        numpy.ndarray: Denoised image with shape (4, H, W), normalized to [0, 1]
    """
    
    # Initialize output array
    H, W = image_4channel.shape[1], image_4channel.shape[2]
    denoised_channels = np.zeros((4, H, W), dtype=np.float32)
    
    # Process each channel separately
    for i in range(4):
        channel = image_4channel[i].astype(np.float32)
        
        # Estimate noise level for this channel
        sigma_est = estimate_sigma(channel)
        print(f"Channel {i+1}: Estimated sigma = {sigma_est:.4f}")
        
        # Convert to PyTorch tensor and add batch and channel dimensions
        channel_tensor = torch.from_numpy(channel).unsqueeze(0).unsqueeze(0).to(device)
        
        # Create the noise level map
        sigma_map = torch.full((1, 1, H, W), sigma_est).to(device)
        
        # Combine image and sigma map
        model_input = torch.cat([channel_tensor, sigma_map], dim=1)
        
        # Denoise the channel
        with torch.no_grad():
            denoised_tensor = model(model_input)
        
        # Convert back to numpy and store
        denoised_channel = denoised_tensor.squeeze().cpu().numpy()
        denoised_channels[i] = np.clip(denoised_channel, 0, 1)
    
    return denoised_channels


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'DPIR/model_zoo/drunet_gray.pth'

n_channels = 1
model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4,
            act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)
_ = model  



# ----------------------------------------------------------
# dark shading estimation on frequency 
# ----------------------------------------------------------
def estimate_dark_shading_differential(noisy_image, dark_frame, mag_threshold=1e-4):
    """
    Estimate dark shading using differential analysis of A and B combinations.
    
    Parameters:
    noisy_image (np.array): Noisy image with arbitrary scene content
    dark_frame (np.array): Dark frame captured under lightless conditions
    n_iter (int): Number of iterations for optimization
    regularization_weight (float): Weight for dark frame prior regularization
    
    Returns:
    np.array: Estimated dark shading
    """
    # Step 1: Create A and B combinations
    A = noisy_image + dark_frame
    B = noisy_image - dark_frame
    
    # Step 2: Compute Fourier transforms
    fft_A = fftpack.fft2(A)
    fft_B = fftpack.fft2(B)
    
    # Step 3: Calculate magnitude and phase spectra
    mag_A = np.abs(fft_A)
    phase_A = np.angle(fft_A)
    
    mag_B = np.abs(fft_B)
    phase_B = np.angle(fft_B)
    
    # Step 4: Analyze the spectral differences
    # Fixed-pattern noise will appear as consistent differences between A and B
    mag_diff = np.abs(mag_A - mag_B)
    phase_diff = np.abs(phase_A - phase_B)
    
    # Step 5: Create a mask for fixed-pattern noise components
    # These are frequencies where the magnitude difference is significant
    # but the phase difference is small (consistent pattern)
    # mag_threshold = np.percentile(mag_diff, 99.)  # Upper quartile
    
    fixed_pattern_mask = (mag_diff > mag_threshold)
    
    # Step 6: Estimate the fixed-pattern noise spectrum
    # Use the average of A and B spectra in the identified regions
    fixed_pattern_spectrum = np.where(
        fixed_pattern_mask,
        (fft_A - fft_B) / 2,  # Average for consistent patterns
        0  # Minimal contribution elsewhere
    )
    
    # Step 7: Convert back to spatial domain
    dark_shading_est = np.real(fftpack.ifft2(fixed_pattern_spectrum))

    return dark_shading_est
    