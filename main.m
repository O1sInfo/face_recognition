clear;
clc;
rng(10);
%% load face dataset.
fprintf("1.------------------Loading face dataset.---------------------\n")
img_dir = "yalefaces/";
subject_id = split("01 02 03 04 05 06 07 08 09 10 11 12 13 14 15");
face_type = split("rightlight leftlight centerlight normal noglasses glasses sad sleepy surprised wink happy");
num_facetypes = length(face_type);
num_classes = length(subject_id);
num_imgs = num_classes * num_facetypes;
% get image size
img_file = strcat(img_dir, "subject", subject_id(1), ".", face_type(1));
img = imread(img_file, 'gif');
[width, height] = size(img);
% create dataset and labels
original_dimention = width*height;
dataSet = zeros(num_imgs, original_dimention);
labels = zeros(num_imgs, 1);
for i = 1:num_classes
    for j = 1:num_facetypes
        img_file = strcat(img_dir, "subject", subject_id(i), ".", face_type(j));
        img = imread(img_file, 'gif');
        k = num_facetypes * (i-1) + j;
        dataSet(k,:) = reshape(img, 1, original_dimention);
        labels(k,:) = i;
    end
end
fprintf("Total face images: %d\n", num_imgs);


%% Split data for training and testing.
fprintf("2.--------Spliting dataset for training and testing.----------\n")
split = 0.2;
num_test_per_subject = round(split*num_facetypes);
sample_idx = zeros(num_classes, num_test_per_subject);
% sample index from each subject for test -- eg: [[4 8 11], [5 10 6], ...]
for i=1:num_classes
    sample_idx(i,:) = randperm(num_facetypes, num_test_per_subject);
    % don't repeat sampling -- randperm.
end
idx = (0:num_classes-1)'*(ones(1, num_test_per_subject).*num_facetypes);
% eg [[0 0 0], [11 11 11], [22 22 22], ...]
testSet_idx = reshape((sample_idx + idx)', 1, num_classes*num_test_per_subject);
% matlab stored at column first so use '
trainSet_idx = setdiff((1:num_imgs), testSet_idx);
% the difference set of sets -- setdiff
testSet = dataSet(testSet_idx,:);
trainSet = dataSet(trainSet_idx,:);
num_trainSet = size(trainSet, 1);
num_testSet = size(testSet, 1);
fprintf("Training num: %d, Testing num %d \n", num_trainSet, num_testSet);


%% Reducing dimension of face data by PCA to extract features.
fprintf("3.-------------Reducing dimension by PCA.---------------------\n")
[coeff, principal_component, latent] = pca(trainSet);
mean_trainSet = mean(trainSet, 1);
center_trainSet = trainSet - repmat(mean_trainSet, num_trainSet, 1);
center_testSet = testSet - repmat(mean_trainSet, num_testSet, 1);
threshold = 0.8;
i = 0;
eigval_sum=0;
while eigval_sum < threshold*sum(latent)
    i=i+1;
    eigval_sum = eigval_sum + latent(i);
end
pca_trainSet = center_trainSet*coeff(:,1:i);
pca_testSet = center_testSet*coeff(:,1:i);
new_dimension = i;
fprintf("Reducing dimension from %d to %d at threshold %0.2f\n",...
    original_dimention, new_dimension, threshold);


fprintf('4. ----Classfication extracted feature by KNN and SVM--------- \n');
%% Classfication for PCA features using KNN
knn_model = fitcknn(pca_trainSet, labels(trainSet_idx, :),'NumNeighbors',5, 'Standardize', 1);
predictClass = predict(knn_model, pca_testSet);
results = labels(testSet_idx, :) == predictClass;
true = sum(results == 1);
accuracy = (true / num_testSet);
fprintf("The Accuracy of prediction using KNN is %0.4f\n", accuracy);

%% Classfication for PCA features using SVM
svm_models = cell(num_classes, 1);
for i=1:num_classes
    svm_models{i} = fitcsvm(pca_trainSet, double(labels(trainSet_idx, :)==i),...
    'Standardize',1,'KernelFunction','RBF',...
    'KernelScale','auto');
end
scores = zeros(num_testSet, num_classes);
for k=1:num_classes
    [~, score] = predict(svm_models{k}, pca_testSet);
    scores(:, k) = score(:,2);
    % Second column contains positive-class scores
end
[~,predictClass] = max(scores,[],2);
results = labels(testSet_idx, :) == predictClass;
true = sum(results == 1);
accuracy = (true / num_testSet);
fprintf("The Accuracy of prediction using SVM is %0.4f\n", accuracy);


%% Inference for some example and show resulting image.
fprintf("5.-----------Inference for some example-----------------------\n")
examlpe_subjects_num = 3;
examlpe_facetypes_num = 3;
example_subjects = randperm(num_classes, examlpe_subjects_num);
examlpe_facetypes = randperm(num_facetypes, examlpe_facetypes_num);
figure(1);
for i=1:examlpe_subjects_num
    for j=1:examlpe_facetypes_num
        subplot(examlpe_subjects_num, examlpe_facetypes_num,(i-1)*examlpe_facetypes_num+j)
        ii = example_subjects(i);
        jj = examlpe_facetypes(j);
        img_file = strcat(img_dir, "subject", subject_id(ii), ".", face_type(jj));
        example_img = imread(img_file, 'gif');
        imshow(example_img)
        test_img = double(reshape(example_img, 1, original_dimention));
        test_img = test_img - repmat(mean_trainSet, 1, 1);
        test_img = test_img*coeff(:,1:new_dimension);
        predict_class = predict(knn_model, test_img);
        title(strcat("subject", subject_id(ii), ".", face_type(jj), '-preict-', num2str(predict_class)))
    end
end

%% Inference for some example and show resulting image.
fprintf("6.-----------Showing some eigenface---------------------------\n")
figure(2);
title('eigenface');
colormap gray
new_dimension_sqrt = ceil(sqrt(new_dimension));
for i=1:new_dimension
    subplot(new_dimension_sqrt,new_dimension_sqrt,i);
    imagesc(reshape(coeff(:,i), width, height))
end
figure(3);
title('reconstructed face');
img_file = strcat(img_dir, "subject", subject_id(1), ".", face_type(1));
example_img = imread(img_file, 'gif');
subplot(221)
imshow(example_img)
test_img = double(reshape(example_img, 1, original_dimention));
test_img_coeff = (test_img - mean_trainSet)*coeff(:,1:new_dimension);
reconstructed_img = test_img_coeff*coeff(:,1:new_dimension)';
colormap gray
subplot(224)
imagesc(reshape(reconstructed_img, width, height))
