clear; clc; close all;

% Setup: Generate synthetic 32x32 image of two closely spaced disks
m = 32;
n = m^2;
[r, c] = ndgrid(1:m, 1:m);
R = 3;
X = double(((r - 16.5).^2 + (c - 13).^2 <= R^2) | ...
    ((r - 16.5).^2 + (c - 20).^2 <= R^2));
x_true = X(:);

% Task 1 Part (a): Construct Gaussian blurring matrix
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

% Task 1 Part (b): Condition number and singular value spectrum
s = svd(A);
kappa = s(1) / s(end);
fprintf('2-norm condition number kappa_2(A) = %.4e\n', kappa);

figure;
semilogy(1:n, s, 'LineWidth', 1.5);
xlabel('Index i');
ylabel('Singular Value \sigma_i');
title('Singular Value Spectrum of Gaussian Blurring Matrix A');
grid on;

% Task 2 Part (a): Failure of naive inversion
b_true = A * x_true;
rng(1);
e0 = randn(n, 1);
e = 0.001 * norm(b_true) * (e0 / norm(e0));
y = b_true + e;
x_naive = A \ y;
rel_err_naive = norm(x_naive - x_true) / norm(x_true);
fprintf('Naive Relative Error E_rel = %.4f\n', rel_err_naive);

% visualize failure of naive inversion
X_naive = reshape(x_naive, m, m);

figure;
subplot(1, 2, 1);
imagesc(X);
axis image off; colormap gray; clim([0 1]);
title('True Image X');

subplot(1, 2, 2);
imagesc(X_naive);
axis image off; colormap gray;
title(sprintf('Naive Reconstruction (E_{rel} = %.1f)', rel_err_naive));


% Task 3 Part (a)
% add Gaussian noise
rng(1);
noise_level = 0.001;
e = randn(n,1);
e = noise_level*norm(b_true)*e/norm(e);
b = b_true + e;

lambdas = [0.1, 0.05, 0.025, 0.01];

figure;

% true image X
subplot(2, 3, 1);
imagesc(X);
axis image off;
colormap gray;
clim([0 1]);
title('True Image X');

% noisy blurred image Y
subplot(2, 3, 2);
imagesc(reshape(y, m, m));
axis image off;
colormap gray;
clim([0 1]);
title('Blurred + Noise Y');

% tikhonov reconstructions for each lambda
for idx = 1:length(lambdas)
    lambda = lambdas(idx);
    x_lambda = (A' * A + lambda^2 * eye(n)) \ (A' * y);

    subplot(2, 3, idx + 2);
    imagesc(reshape(x_lambda, m, m));
    axis image off;
    colormap gray;
    clim([0 1]);
    title(sprintf('Restored (\\lambda = %.3g)', lambda));
end

% Task 4 Part (a): Parameter sweep
lambdas = logspace(-5, 0, 25);
rel_errors = zeros(size(lambdas));

% compute A'*A and A'*y outside the loop 
AtA = A' * A;
Aty = A' * y;
I_n = eye(n);
norm_x_true = norm(x_true);

for idx = 1:length(lambdas)
    lam = lambdas(idx);
    x_lam = (AtA + lam^2 * I_n) \ Aty;
    rel_errors(idx) = norm(x_lam - x_true) / norm_x_true;
end

% identify the optimal lambda that minimizes E_rel
[min_err, best_idx] = min(rel_errors);
lambda_star = lambdas(best_idx);
fprintf('Optimal lambda* = %.4e (Grid index %d)\n', lambda_star, best_idx);
fprintf('Minimum relative error E_rel(lambda*) = %.6f\n', min_err);

% compute the optimal reconstruction X_lambda
x_best = (AtA + lambda_star^2 * I_n) \ Aty;
X_best = reshape(x_best, m, m);

% plot error curve alongside optimal reconstruction
figure;

subplot(1, 2, 1);
loglog(lambdas, rel_errors, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 5); hold on;
loglog(lambda_star, min_err, 'rp', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
xlabel('Regularization Parameter \lambda');
ylabel('Relative Reconstruction Error E_{rel}(\lambda)');
title('Tikhonov Error Curve (Log-Log Scale)');
legend('E_{rel}(\lambda)', sprintf('Optimal \\lambda^* = %.2e', lambda_star), ...
    'Location', 'northeast');
grid on;

subplot(1, 2, 2);
imagesc(X_best);
axis image off;
colormap gray;
colorbar;
title(sprintf('Optimal Reconstruction (\\lambda^* = %.2e)', lambda_star));