function [Regions, Means, Stds] = feature_extractor(image_filepaths, segmentation_filepaths)

opts = detectImportOptions('FreeSurferColorLUT.txt');
opts.VariableNames = {'Index', 'Name', 'R', 'G', 'B', 'A'};
opts.CommentStyle = '#';
labels = readtable("FreeSurferColorLUT.txt", opts);

indices = labels.Index;

for i = 1:1:length(image_filepaths)

    segmentation = niftiread(string(segmentation_filepaths(i)));
    pixel_values = unique(segmentation);
    indices = intersect(indices, pixel_values);

end

[~, pos] = intersect(labels.Index, indices);
Regions = string(labels.Name(pos));

Means = [NaN; indices];
Stds = [NaN; indices];

for i = 1:1:length(image_filepaths)
    
    image = niftiread(string(image_filepaths(i)));
    segmentation = niftiread(string(segmentation_filepaths(i)));

    f_mean = [i];5
    f_std = [i];

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
