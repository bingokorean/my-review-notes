%% PCA
% training images
n = 20;
x = zeros(n, 45*40); %size(x) = [ 20 x 1800 ] 
for k=1:n,
    fname = sprintf('%d.png',k);
    img = double(rgb2gray(imread(fname)));
    x(k,:) = (img(:))';
end;

% pca
c = cov(x); % returns covariance matrix (symmetric matrix)
[v, d] = eig(c); % returns the collection of eigenvectors, V and diagonal matrix D of eigenvalues

%size(c) % [ 1800 x 1800 ]
%size(v) % [ 1800 x 1800 ]
%size(d) % [ 1800 x 1800 ]

% average face
face = zeros(45,40);
face(:) = mean(x);
avg = mean(x);
imwrite(uint8(face), 'avg.png');

% eigenfaces (first 20 faces)
for k=1:20,
    fname = sprintf('eig%d.png',k);
    face(:) = v(:,45*40-k+1);
    imwrite(uint8(face/max(face(:))*255), fname);
end;

% reconstruction using only k eigenfaces
cnt = [20, 10, 5, 2];
for k=cnt,
    % K-L transform
    v_k = v(:,45*40-k+1:45*40);
    %size(v_k) = [1800 x 20] (k=20일 때),  [1800 x 10],  [1800 x 5],  [1800 x 2] 
    y_k = x*v_k;
    %size(y_k) = [20 x 20],   [20 x 10],   [20 x 5],   [20 x 2]
    % reconstruction
    x_recons = v_k*y_k'; % [1800 x 20]
    for i=1:n,
        fname = sprintf('%dres%d.png',i,k);
        face(:) = x_recons(:,i);
        imwrite(uint8(face), fname);
    end;
end;



%% (Assignment) Reconstruction of glass image
% 1. training images
x = zeros(1, 45*40); %size(x) = [ 1 x 1800 ] 
flt_x = zeros(45*40, 1); % [ 1800 x 1 ]

fname = sprintf('glass_1_before.png');
img = double(rgb2gray(imread(fname)));
x(1,:) = (img(:))';


% 2. reconstruction using only k eigenfaces
cont = [20];
for k=cont,
    % K-L transform
    v_k = v(:,45*40-k+1:45*40);
    %size(v_k) = [1800 x 20] (k=20일 때),  [1800 x 10],  [1800 x 5],  [1800 x 2] 
    y_k = x*v_k;
    %size(y_k) = [20 x 20],   [20 x 10],   [20 x 5],   [20 x 2]
    % reconstruction
    x_recons = v_k*y_k'; % size(x_recons) = [1800 x 1]

    fname = sprintf('glass_1_after_eig%d.png', k);
    face(:) = x_recons(:,1); % [ 1800 x 1 ]
    %size(face) = [ 45 x 40 ]
    imwrite(uint8(face), fname);
end;



origin = x(1,:)'; % [ 1800 x 1 ]
avg_x = avg'; 
% 3. difference between original and reconst
for i=1:1800
    if abs(origin(i,1) - x_recons(i, 1)) > 95
        flt_x(i, 1) = 255;
        origin(i, 1) = avg_x(i, 1);
    else
        flt_x(i, 1) = 0;
    end
end


fname = sprintf('glass_binary.png');
face(:) = flt_x(:,1); 
imwrite(uint8(face), fname);


fname = sprintf('glass_1_final.png');
face(:) = origin(:,1); 
imwrite(uint8(face), fname);
















