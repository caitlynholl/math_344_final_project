clear; clc; close all;

% Step 1: Read original H image
H = imread('original_H.png');
if size(H,3) == 3
    H = rgb2gray(H);
end
H = im2double(H);
H = imresize(H,[32 32]);
x_true = H(:);
m = 32;
n = m^2;

% Step 2: Construct Gaussian blurring matrix
sigma = 2;
radius = 6;
[i,j] = meshgrid(-radius:radius);
G = exp(-(i.^2+j.^2)/(2*sigma^2));
G = G/sum(G(:));
A = zeros(n,n);
for k = 1:n
    E = zeros(m,m);
    E(k) = 1;
    B = conv2(E,G,'same');
    A(:,k) = B(:);
end

% Step 3: Blur original image
b_true = A*x_true;

% Step 4: Add Gaussian noise
rng(1);
noise_level = 0.001;
e = randn(n,1);
e = noise_level*norm(b_true)*e/norm(e);
b = b_true + e;

% Step 5: Tikhonov Regularization
lambda = 0.05;
x_lambda = (A'*A + lambda^2*eye(n)) \ (A'*b);

% Step 6: Display images
figure;
subplot(1,4,1);
imagesc(H);
axis image off;
colormap gray;
caxis([0 1]);
title('Original H');

subplot(1,4,2);
imagesc(reshape(b_true,m,m));
axis image off;
caxis([0 1]);
title('Blurred H');

subplot(1,4,3);
imagesc(reshape(b,m,m));
axis image off;
caxis([0 1]);
title('Blurred H + Noise');

subplot(1,4,4);
imagesc(reshape(x_lambda,m,m));
axis image off;
caxis([0 1]);
title('Restored H');

% Step 7: Relative reconstruction error
relative_error = norm(x_lambda-x_true)/norm(x_true);
fprintf('Relative Reconstruction Error = %.6f\n',relative_error);