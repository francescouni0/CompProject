clear all
close all
clc

%% Visualizzazione (Non necessario se importo da Python)
image_filename = "C:\Users\simol\OneDrive\Documenti\GitHub\CompProject\Diffusion_parameters_maps-20230215T134959Z-001\Diffusion_parameters_maps\003_S_4152\corrected_AD_image\2011-08-30_09_55_00.0\I359077\ADNI_003_S_4152_MR_corrected_AD_image_Br_20130213172802294_S120805_I359077.nii";
mask_filename = "C:\Users\simol\OneDrive\Documenti\GitHub\CompProject\Diffusion_space_segmentations-20230215T134839Z-001\Diffusion_space_segmentations\003_S_4152_wmparc_on_MD.nii.gz";

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
figure('Name', 'DTI, central slice')
imagesc(image(:,:,central_slice))
% colormap("gray")
figure('Name', 'Mask, central slice')
imagesc(segmentation(:,:,central_slice))
xlabel('Columns')
ylabel('Rows')

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
image_filepaths = ["C:\Users\simol\OneDrive\Documenti\GitHub\CompProject\Diffusion_parameters_maps-20230215T134959Z-001\Diffusion_parameters_maps\003_S_4081\corrected_AD_image\ADNI_003_S_4081_MR_corrected_AD_image_Br_20130213172736437_S114205_I359073.nii", "C:\Users\simol\OneDrive\Documenti\GitHub\CompProject\Diffusion_parameters_maps-20230215T134959Z-001\Diffusion_parameters_maps\003_S_4119\corrected_AD_image\ADNI_003_S_4119_MR_corrected_AD_image_Br_20130213172743850_S118964_I359074.nii"];
segmentation_filepaths = ["C:\Users\simol\OneDrive\Documenti\GitHub\CompProject\Diffusion_space_segmentations-20230215T134839Z-001\Diffusion_space_segmentations\003_S_4081_wmparc_on_MD.nii.gz", "C:\Users\simol\OneDrive\Documenti\GitHub\CompProject\Diffusion_space_segmentations-20230215T134839Z-001\Diffusion_space_segmentations\003_S_4119_wmparc_on_MD.nii.gz"];

Features = feature_extractor(image_filepaths, segmentation_filepaths)

a = Features(1,3)

%% 
A = [0 1 2 3];
B = [4 5 6 7 8];
C = [9];

patient(1).name = 'Alpha';
patient(1).value = A;
patient(2).name = 'Beta';
patient(2).value = B;
patient(3).name = 'Omega';
patient(3).value = C;

%% 
for i = 1:1:10
    feature(i).subject = ['Name ', num2str(i)];
end

