clear all
close all
clc

%% Visualizzazione (Non necessario se importo da Python)
image_filename = 'Diffusion_parameters_maps-20230215T134959Z-001\\Diffusion_parameters_maps\\098_S_4002\\corrected_FA_image\\2011-02-28_15_42_50.0\\I397180\\ADNI_098_S_4002_MR_corrected_FA_image_Br_20131105134057196_S100616_I397180.nii';
mask_filename = 'Diffusion_space_segmentations-20230215T134839Z-001\\Diffusion_space_segmentations\\098_S_4002_wmparc_on_MD.nii.gz';

% Leggo le info relative all'immagine NIFTI
image_info = niftiinfo(image_filename);
mask_info = niftiinfo(mask_filename);
 
% Importo l'immagine NIFTI
image = niftiread(image_filename);
segmentation = niftiread(mask_filename);

% Mostro in sequenza tutte le slice della DTI
figure('Name', 'DTI, all slices')
montage(image,'Indices',1:1:size(image,3),'DisplayRange',[])

%Mostro le slice centrali delle immagini NIFTI
central_slice=round(size(image,3)/2);
k_slice=40;

segmentation(segmentation~=0 & segmentation~=18 & segmentation~=54)=1;
segmentation(segmentation==18 | segmentation==54)=0.5;

figure('Name', 'DTI, central slice')
imagesc(image(:,:,k_slice))
% colormap("gray")
figure('Name', 'Mask, central slice')
imagesc(segmentation(:,:,k_slice))
xlabel('Columns')
ylabel('Rows')

for i=44:46
    figure(i)
    subplot(1,2,1)
    imagesc(image(:,:,i))
    subplot(1,2,2)
    imagesc(segmentation(:,:,i))
end

%% Segmentazione
% Importo la tabella di conversione Indice-Regione
opts = detectImportOptions('FreeSurferColorLUT.txt');
opts.VariableNames = {'Index', 'Name', 'R', 'G', 'B', 'A'};
opts.CommentStyle = '#';
labels = readtable("FreeSurferColorLUT.txt", opts);

% Identifico gli indici presenti nell'immagine
indices = unique(segmentation);

% Per ogni indice realizzo una maschera booleana
bool_mask = segmentation == indices(181);
% bool_mask = segmentation == 5002;

% Creo un'immagine con i soli pixel classificati con il k-esimo indice
mask_segment = segmentation.*bool_mask;
image_segment = image.*bool_mask;

% Test: verifico il corretto funzionamento selezionando lo stesso punto
% nell'immagine originale e in quella segmentata e osservando che i valori
% siano concordi
figure('Name', 'Test on segmentation, central slice')
imagesc(mask_segment(:,:,central_slice));
figure('Name', 'Test on image, central slice')
imagesc(image_segment(:,:,central_slice));

% Estraggo dall'immagine originale i valori dei pixel corrispondenti al
% k-esimo indice
results = image(bool_mask);
mean_val = mean(results)
std_val = std(results)

%% Estrazione delle features
[val, pos] = intersect(labels.Index, indices);
Region = string(labels.Name(pos));
f_mean = [];
f_std = [];

for i = 1:1:length(indices)
    bool_mask = segmentation == indices(i);
    results = image(bool_mask);
    f_mean = [f_mean mean(results)];
    f_std = [f_std std(results)];
end

Features = table(Region, f_mean.', f_std.');
Features.Properties.VariableNames = ["Region" "Mean" "Standard deviation"];

%% 
test = table('Size', [0,4], ...
    'VariableTypes', ["double", "string", "cell", "cell"], ...
    'VariableNames', ["Subject", "Region", "Mean", "Standard Deviation"])

row = {1, Region.', f_mean, f_std}

% test(1, :) = {a, b, c, d}

test = [test; row]
test

%% 
test = table('Size', [0,4], ...
    'VariableTypes', ["double", "string", "cell", "cell"], ...
    'VariableNames', ["Subject", "Region", "Mean", "Standard Deviation"])

row = table(1, Region.', f_mean, f_std, ...
    'VariableNames', ["Subject", "Region", "Mean", "Standard Deviation"])

% test_new = [test; row]

%% 
test = table;

for i = 1:1:5
    row = {i, Region.', f_mean, f_std}
    test = [test; row]
end

test.Properties.VariableNames = ["A", "B", "C", "D"]

%% 
image_filepaths = ["\\?\C:\Users\simol\OneDrive\Documenti\GitHub\CompProject\Diffusion_parameters_maps-20230215T134959Z-001\Diffusion_parameters_maps\098_S_4002\corrected_FA_image\2011-02-28_15_42_50.0\I397180\ADNI_098_S_4002_MR_corrected_FA_image_Br_20131105134057196_S100616_I397180.nii", "\\?\C:\Users\simol\OneDrive\Documenti\GitHub\CompProject\Diffusion_parameters_maps-20230215T134959Z-001\Diffusion_parameters_maps\098_S_4003\corrected_FA_image\2011-03-22_09_23_47.0\I299742\ADNI_098_S_4003_MR_corrected_FA_image_Br_20120421215950180_S102157_I299742.nii"];
segmentation_filepaths = ["C:\Users\simol\OneDrive\Documenti\GitHub\CompProject\Diffusion_space_segmentations-20230215T134839Z-001\Diffusion_space_segmentations\098_S_4002_wmparc_on_MD.nii.gz", "C:\Users\simol\OneDrive\Documenti\GitHub\CompProject\Diffusion_space_segmentations-20230215T134839Z-001\Diffusion_space_segmentations\098_S_4003_wmparc_on_MD.nii.gz"];

% [ro, mo, so] = feature_extractor_old(image_filepaths, segmentation_filepaths);
[r, m, s] = feature_extractor(image_filepaths, segmentation_filepaths);

%% 
A = [0 1 2 3].';
B = [4 5 6 7 8].';
C = [9];
D = [2 4 6 8].';
E = ["a" "b" "c" "d"].';

if length(A)==length(E)
    R = [A E]
else
    disp('Caca')
end

%% 
a = {};

for i=1:10
    for j=1:5
        a{i,j}=[i j];
    end
end

%% 
n=10;
priors=struct();
for ind=1:n
    priors.("Subject_"+ind) = ind;
end

%% 
list = [];
for i = 1:10
    list = [list i];
end
list = [list 1 2 3]
list = unique(list)

%% 
image_filepaths = ["\\?\C:\Users\simol\OneDrive\Documenti\GitHub\CompProject\Diffusion_parameters_maps-20230215T134959Z-001\Diffusion_parameters_maps\003_S_4081\corrected_FA_image\ADNI_003_S_4081_MR_corrected_FA_image_Br_20120421203936832_S114205_I299596.nii", "\\?\C:\Users\simol\OneDrive\Documenti\GitHub\CompProject\Diffusion_parameters_maps-20230215T134959Z-001\Diffusion_parameters_maps\098_S_4003\corrected_FA_image\2011-03-22_09_23_47.0\I299742\ADNI_098_S_4003_MR_corrected_FA_image_Br_20120421215950180_S102157_I299742.nii"];

% Importo l'immagine NIFTI
image = niftiread(image_filepaths(1));
central_slice = image(:,:,30);
central_slice(central_slice == 0) = NaN;

% Mostro in sequenza tutte le slice della DTI
figure('Name', 'DTI, all slices')
montage(image,'Indices',1:1:size(image,3),'DisplayRange',[])

figure('Name', 'DTI, central slice')
imagesc(central_slice)

