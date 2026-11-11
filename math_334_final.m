clc; clear; close all;

% load image and downsample to 64x64 so svd(A) fits in RAM
img_raw = imread('/Users/caitlynholl/Downloads/xgTik.png');
if size(img_raw, 3) == 3, img_raw = rgb2gray(img_raw); end
img_orig = imresize(im2double(img_raw), [64 64]); 
[m, n] = size(img_orig);
num_pixels = m * n;
x = img_orig(:);

% create Gaussian point-spread function
sz = 5; sigma_psf = 1.5;
[xg, yg] = meshgrid(-(sz-1)/2:(sz-1)/2, -(sz-1)/2:(sz-1)/2);
h = exp(-(xg.^2 + yg.^2) / (2 * sigma_psf^2));
h = h / sum(h(:));

% construct explicit matrix operator A (num_pixels x num_pixels)
A = construct_explicit_A(h, m, n);

% add noise to observed image
noise_level = 0.02;
e = noise_level * randn(num_pixels, 1);
y = A * x + e;
Y_observed = reshape(y, m, n);

% SVD computation
[U, S, V] = svd(A);
sigma = diag(S);
Ut_y = U' * y;

% sweep through a range of lambda values
lambda_vec = logspace(-4, 0, 50);
err_vec = zeros(size(lambda_vec));

for k = 1:length(lambda_vec)
    l = lambda_vec(k);
    f_i = (sigma.^2) ./ (sigma.^2 + l^2);
    x_l = V * ( (f_i .* Ut_y) ./ sigma );
    err_vec(k) = norm(x_l - x) / norm(x);
end

[min_err, opt_idx] = min(err_vec);
opt_lambda = lambda_vec(opt_idx);
fprintf('Optimal lambda: %.4f, with minimum relative error: %.4f\n', opt_lambda, min_err);

% Reconstruct optimal image
f_opt = (sigma.^2) ./ (sigma.^2 + opt_lambda^2);
x_opt = V * ( (f_opt .* Ut_y) ./ sigma );
X_opt = reshape(x_opt, m, n);

% Visualization
figure('Position', [100 100 1200 400]);
subplot(1,4,1); imshow(img_orig); title('Original Image X');
subplot(1,4,2); imshow(Y_observed); title('Blurred & Noisy y');
subplot(1,4,3); imshow(X_opt); title(sprintf('Reconstructed (\\lambda = %.4f)', opt_lambda));
subplot(1,4,4); semilogx(lambda_vec, err_vec, 'b-', 'LineWidth', 1.5); hold on;
plot(opt_lambda, min_err, 'ro', 'MarkerSize', 8, 'LineWidth', 2);
grid on; xlabel('\lambda'); ylabel('Relative Error E_{rel}(\lambda)');
title('Error vs. Regularization Parameter');

% Helper function to construct matrix A from Gaussian PSF
function A = construct_explicit_A(psf, m, n)
N = m * n;
pad_psf = zeros(m, n);
pad_psf(1:size(psf,1), 1:size(psf,2)) = psf;
pad_psf = circshift(pad_psf, -[floor(size(psf,1)/2), floor(size(psf,2)/2)]);
F = fft2(pad_psf);

A = zeros(N, N);
for col = 1:N
    e = zeros(N, 1);
    e(col) = 1;
    E = reshape(e, m, n);
    conv_col = real(ifft2(fft2(E) .* F));
    A(:, col) = conv_col(:);
end
end