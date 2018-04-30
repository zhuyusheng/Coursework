
clear;





%% Face Detection
faceDetect = vision.CascadeObjectDetector;


%% Preprocessing
files = dir('jaffe/*.tiff');

j = 1; %index for creating the matrix
for file = files'
       pic = imread(['jaffe/',file.name]);

       %convert to grayscale
       if size(pic,3) == 3
           pic = rgb2gray(pic);
       end
       
       %face detection fails on a few images so using "try" to avoid errors
       %ending the script
       try 
           %face detection

           region = step(faceDetect, pic);

           %crop face
           pic = imcrop(pic, region);
           
           %resize
           pic = imresize(pic, [300,300]);
           
           %# Create an ellipse shaped mask
           c = fix(size(pic) / 2);   %# Ellipse center point (y, x)
           r_sq = [120, 150] .^ 2;  %# Ellipse radii squared (y-axis, x-axis)
           [X, Y] = meshgrid(1:size(pic, 2), 1:size(pic, 1));
           ellipse_mask = (r_sq(2) * (X - c(2)) .^ 2 + ...
               r_sq(1) * (Y - c(1)) .^ 2 <= prod(r_sq));
           pic = bsxfun(@times, pic, uint8(ellipse_mask)); %# Apply the mask to the image
           
           %histogram equalization
           pic = histeq(pic);
           

           
           %downsample
           %pic = imresize(pic, 1/2);
                  
           %save rescaled pic size
           image_dims = size(pic);
            
           %write to the matrix as a column vector
           imageMatrix(:,j) = reshape(pic,[],1);
           
           %save the image index numbers that were processed
           picNums(j) = i;
           
           j = j + 1;          
           
       end
end

%% %convert imageMatrix to Double

imageMatrix = double(imageMatrix);

%cleanup
clear i j pic picFile

%% Mean Face
meanFace = mean(imageMatrix,2);

meanFace = reshape(meanFace, image_dims);

meanFace = uint8(meanFace);

imshow(meanFace);

%% find the mean face
meanFace = mean(imageMatrix,2);

%create the mean shifted images
shifted_images = imageMatrix - repmat(meanFace, 1, size(imageMatrix,2));

% calculate the ordered eigenvectors and eigenvalues
[evectors, score, evalues] = pca(imageMatrix');

%% plot to determine how many eigenvector to use.
% display the eigenvalues

normalised_evalues = evalues / sum(evalues);
figure, plot(cumsum(normalised_evalues));
xlabel('No. of eigenvectors'), ylabel('Variance accounted for');
xlim([1 213]), ylim([0 1]), grid on;
title("Cumulative Percent Variance Explained by Component");

%% only retain the top 'num_eigenfaces' eigenvectors (i.e. the principal components)
num_eigenfaces = 60;
evectors = evectors(:, 1:num_eigenfaces);

%% project the images into the subspace to generate the feature vectors
features = evectors' * shifted_images;


%% Show the eigenfaces
% display the eigenvectors

eig0 = reshape(meanFace, image_dims);
figure,subplot(4,4,1)
imagesc(eig0)
colormap gray
for i = 1:15
    subplot(4,4,i+1)
    imagesc(reshape(evectors(:,i), image_dims))
end

figure
colormap gray
for i = 1:4
    subplot(2,2,i)
    imagesc(reshape(evectors(:,i), image_dims))
    title(strcat('Eigenvector ', num2str(i)))
end

%% individual eigenfaces

colormap gray
imagesc(reshape(evectors(:,2), image_dims))
%% mean face

colormap gray
imagesc(eig0)

%% features
features2 = features';


emotionTable_j = struct2table(files);
emotionTable_j = emotionTable_j(:,1);

%%
clear emotion
i = 1;
for file = files'
    x = file.name;
    x = strsplit(x, '.'); 
    x = x(2);
    emotion(i) = x;
    i = i + 1;
    
end

emotion = char(emotion);
emotion = emotion(:,1:2);
emotion = table(emotion);

%% writing files
csvwrite('features_j.csv',features2);

%% writing emotion list
writetable(emotion, 'emotionTable_j.txt');

