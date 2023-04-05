function [Regions, Means, Stds] = feature_extractor_par(image_filepaths, segmentation_filepaths)
% FEATURE_EXTRACTOR: Extract mean and variance from the (common) regions of 
%                    interest specified in the segmentation maps. Uses
%                    "parfor" to parallelize the exctraction process over
%                    the CPU cores.
%
% Args:
%
%   image_filepaths(string): 
%       String array storing the paths to the NIFTI images (diffusion 
%       parameters maps) you desire to extract the features from.
%
%   segmentation_filepaths(string):
%       String array storing the paths to the NIFTI-GZ files (diffusion
%       space segmentations) containing the ROIs from which to extract the
%       features.
%
% Returns:
%
%   Regions(string):
%       String array storing the common ROIs present in the segmentation 
%       maps provided.
%
%   Means(single):
%       Array storing the mean of pixel's values for each ROI. Rows are for
%       regions, columns for subjects. First row reports the subject index,
%       while the firs column reports the ROI index according to the
%       FreeSurfer Color LUT.
%
%   Stds(single):
%       Array storing the standard deviation of pixel's values for each 
%       ROI. Rows are for regions, columns for subjects. First row reports 
%       the subject index, while the firs column reports the ROI index 
%       according to the FreeSurfer Color LUT.
%
% See also READTABLE, NIFTIREAD, UNIQUE, INTERSECT, PARFOR.


% Set options to import the FreeSurferColorLUT.txt file
opts = detectImportOptions('FreeSurferColorLUT.txt');
opts.VariableNames = {'Index', 'Name', 'R', 'G', 'B', 'A'};
opts.CommentStyle = '#';
% Import FreeSurferColorLUT.txt
labels = readtable("FreeSurferColorLUT.txt", opts);

% Store the ROI's numerical indices in an array
indices = labels.Index;

% Find the common indeces present in all segmentation maps, use parfor to
% parallelize the process
parfor i = 1:1:length(image_filepaths)

    segmentation = niftiread(string(segmentation_filepaths(i)));
    pixel_values = unique(segmentation);
    indices = intersect(indices, pixel_values);

end

% Store in an array the positions in labels.Index array of the common
% indices
[~, pos] = intersect(labels.Index, indices);
% Use the position values mentioned above to extract from labels.Name the
% names of the common ROIs
Regions = string(labels.Name(pos));

Means = [NaN; indices];
Stds = [NaN; indices];

% Loop over all images, use parfor to parallelize the process
parfor i = 1:length(image_filepaths)
    
    image = niftiread(string(image_filepaths(i)));
    segmentation = niftiread(string(segmentation_filepaths(i)));

    f_mean = [i];
    f_std = [i];

% Loop over all indeces to create a bool mask and extract the corrisponding
% ROI's values
    for j = 1:1:length(indices)

        bool_mask = segmentation == indices(j);
        results = image(bool_mask);
        f_mean = [f_mean; mean(results)];
        f_std = [f_std; std(results)];

    end

Means = [Means f_mean];
Stds = [Stds f_std];

end

end
