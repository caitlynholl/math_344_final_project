% Math 334: gradient-penalty Tikhonov denoising (base MATLAB).
% Run this script; upload the generated PNGs and TEX table to Overleaf.
clear; close all; clc;
rng(2026, 'twister');
root = fileparts(mfilename('fullpath'));
if isempty(root), root = pwd; end
dataDir = fullfile(root, 'images');
outDir = fullfile(root, 'results');
if ~exist(dataDir, 'dir'), mkdir(dataDir); end
if ~exist(outDir, 'dir'), mkdir(outDir); end
names = {'cameraman', 'barbara'};
ids = {'01', '09'};
noiseSD = 25/255;
lambdas = [0.2, 0.7, 1.5];
rows = cell(0, 5);

for k = 1:numel(names)
    imagePath = fullfile(dataDir, [names{k} '.png']);
    if ~exist(imagePath, 'file')
        url = ['https://raw.githubusercontent.com/cszn/DnCNN/master/' ...
               'testsets/Set12/' ids{k} '.png'];
        websave(imagePath, url);
    end
    X = double(imread(imagePath))/255;
    % Barbara is 512x512: average each 2x2 block BEFORE adding noise.
    if isequal(size(X), [512, 512])
        X = (X(1:2:end,1:2:end) + X(2:2:end,1:2:end) + ...
             X(1:2:end,2:2:end) + X(2:2:end,2:2:end))/4;
    end
    [h,w] = size(X);
    xTrue = X(:);  % MATLAB column-major vectorization.
    Y = X + noiseSD*randn(h,w);  % Do not clip the observations.
    y = Y(:);

    % Forward differences only between adjacent pixels INSIDE the image.
    % No wrap-around; this gives natural (zero-flux) boundary conditions.
    Dr = sparse(1:h-1,1:h-1,-ones(1,h-1),h-1,h) + ...
         sparse(1:h-1,2:h,ones(1,h-1),h-1,h);
    Dc = sparse(1:w-1,1:w-1,-ones(1,w-1),w-1,w) + ...
         sparse(1:w-1,2:w,ones(1,w-1),w-1,w);
    Drow = kron(speye(w),Dr);
    Dcol = kron(Dc,speye(h));
    G = Drow'*Drow + Dcol'*Dcol;
    I = speye(h*w);

    mse = mean((y-xTrue).^2);
    rows(end+1,:) = {names{k}, 'Noisy', 0, ...
                    norm(y-xTrue)/norm(xTrue), 10*log10(1/mse)};
    panels = {X,Y};
    labels = {'Original','Noisy'};
    for j = 1:numel(lambdas)
        lambda = lambdas(j);
        % Solve the sparse system; do not explicitly compute an inverse.
        xHat = (I + lambda^2*G) \ y;
        mse = mean((xHat-xTrue).^2);
        rows(end+1,:) = {names{k}, 'Tikhonov', lambda, ...
            norm(xHat-xTrue)/norm(xTrue), 10*log10(1/mse)};
        panels{end+1} = reshape(xHat,h,w);
        labels{end+1} = sprintf('lambda = %.1f',lambda);
    end

    fig = figure('Color','w','Position',[50,100,1500,360]);
    for j = 1:numel(panels)
        subplot(1,numel(panels),j);
        imagesc(panels{j},[0,1]); axis image off;
        title(labels{j});
    end
    colormap(fig,gray(256));
    set(fig,'PaperPositionMode','auto');
    print(fig,fullfile(outDir,[names{k} '_comparison.png']),'-dpng','-r200');
end

T = cell2table(rows,'VariableNames', ...
    {'Image','Method','Lambda','RelativeError','PSNR_dB'});
disp(T);
writetable(T,fullfile(outDir,'denoising_metrics.csv'));

% Generate a LaTeX table containing the ACTUAL MATLAB results.
fid = fopen(fullfile(outDir,'denoising_metrics.tex'),'w');
assert(fid ~= -1,'Could not create the LaTeX table.');
fprintf(fid,'%s\n','\begin{table}[htbp]');
fprintf(fid,'%s\n','\centering');
fprintf(fid,'%s\n','\caption{Denoising results with noise standard deviation $25/255$. Metrics use unclipped pixel values.}');
fprintf(fid,'%s\n','\label{tab:denoising-results}');
fprintf(fid,'%s\n','\begin{tabular}{llrrr}');
fprintf(fid,'%s\n','\hline');
fprintf(fid,'%s\n','Image & Method & $\lambda$ & Relative error & PSNR (dB) \\');
fprintf(fid,'%s\n','\hline');
for r = 1:size(rows,1)
    if strcmp(rows{r,2},'Noisy'), lam = '--';
    else, lam = sprintf('%.1f',rows{r,3}); end
    line = sprintf('%s & %s & %s & %.4f & %.2f', ...
        rows{r,1},rows{r,2},lam,rows{r,4},rows{r,5});
    fprintf(fid,'%s\n',[line ' \\']);
end
fprintf(fid,'%s\n','\hline');
fprintf(fid,'%s\n','\end{tabular}');
fprintf(fid,'%s\n','\end{table}');
fclose(fid);
fprintf('\nOutputs saved in: %s\n',outDir);
