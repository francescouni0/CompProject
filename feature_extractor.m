function [Features] = feature_extractor(image_filepaths, segmentation_filepaths)

opts = detectImportOptions('FreeSurferColorLUT.txt');
opts.VariableNames = {'Index', 'Name', 'R', 'G', 'B', 'A'};
opts.CommentStyle = '#';
labels = readtable("FreeSurferColorLUT.txt", opts);

Features = table;

for i = 1:1:length(image_filepaths)
    
    image = niftiread(image_filepaths(i));
    segmentation = niftiread(segmentation_filepaths(i));

    indices = unique(segmentation);

    [~, pos] = intersect(labels.Index, indices);
    region = string(labels.Name(pos));
    f_mean = [];
    f_std = [];

    for j = 1:1:length(indices)
        bool_mask = segmentation == indices(j);
        results = image(bool_mask);
        f_mean = [f_mean mean(results)];
        f_std = [f_std std(results)];
    end

    row = {i, region.', f_mean, f_std};
    Features = [Features; row];

end

Features.Properties.VariableNames = ["Subject", "Region", "Mean", "Standard Deviation"];

end