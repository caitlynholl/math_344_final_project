clear; clc; close all;

% Step 1: Read original color image
H = im2double(imread('hubble_2.jpeg'));
H = imresize(H,[32 32]);

if size(H,3) == 1
    H = repmat(H,[1 1 3]);
end

m = 32;
n = m^2;
numChannels = 3;

% Step 2: Construct Gaussian blurring matrix
sigma = 2;
radius = 6;
[i,j] = meshgrid(-radius:radius);
G = exp(-(i.^2 + j.^2)/(2*sigma^2));
G = G/sum(G(:));

A = zeros(n,n);
for k = 1:n
    E = zeros(m,m);
    E(k) = 1;
    B = conv2(E,G,'same');
    A(:,k) = B(:);
end

% Step 3: Blur original image channel-by-channel
x_true = reshape(H,[],numChannels);
b_true = zeros(m,m,numChannels);
for c = 1:numChannels
    b_true(:,:,c) = reshape(A*x_true(:,c),m,m);
end

% Step 4: Add Gaussian noise channel-by-channel
rng(1);
noise_level = 0.001;
b = zeros(m,m,numChannels);
for c = 1:numChannels
    e = randn(n,1);
    e = noise_level*norm(b_true(:,:,c),'fro')*e/norm(e);
    b(:,:,c) = reshape(reshape(b_true(:,:,c),[],1) + e,m,m);
end

% Step 5: Tikhonov regularization channel-by-channel
lambda = 0.01;
x_lambda = zeros(n,numChannels);
for c = 1:numChannels
    x_lambda(:,c) = (A'*A + lambda^2*eye(n)) \ (A'*reshape(b(:,:,c),[],1));
end
x_lambda = reshape(x_lambda,m,m,numChannels);

% Step 6: Display images
figure;
subplot(1,4,1);
imshow(H);
title('Original');

subplot(1,4,2);
imshow(b_true);
title('Blurred');

subplot(1,4,3);
imshow(b);
title('Blurred + Noise');

subplot(1,4,4);
imshow(x_lambda);
title('Restored');

% Step 7: Relative reconstruction error
relative_error = norm(x_lambda(:) - H(:)) / norm(H(:));
fprintf('Relative Reconstruction Error = %.6f\n',relative_error);
