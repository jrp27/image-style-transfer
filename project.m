% requires base_image to be a square and style_image to be equal to or
% larger in size than base_image along both dimensions
style_image = imread('watercolor.jpg');
base_image = imread('kitten.jpg');
global totalWidth;
totalWidth = size(base_image, 1);

% step 1: split and match
min_width = 4;
max_width = 256;
omega = 15;
K = 1;

% regions is cell array where each row contains: patch, xOffset, yOffset, xCenter, yCenter, width;
regions = {base_image, 1, 1, size(base_image, 1)/2, size(base_image, 2)/2, size(base_image,1)};
global PATCH_INDEX;
PATCH_INDEX = 1;
global X_OFFSET_INDEX;
X_OFFSET_INDEX = 2;
global Y_OFFSET_INDEX;
Y_OFFSET_INDEX = 3;
global X_CENTER_INDEX;
X_CENTER_INDEX = 4;
global Y_CENTER_INDEX;
Y_CENTER_INDEX = 5;
global WIDTH_INDEX;
WIDTH_INDEX = 6;

% candidate labels is cell array where each row has 2 entries: index into
% regions for corresponding region, cell array containing: style 
% patch, distance b/w region patch and style patch, centerX of style patch,
% centerY of style patch
candidateLabels = {};

index = 1;
done = false;

while not(done)
    % R_i
     patch = regions{index, PATCH_INDEX};
     width = regions{index, WIDTH_INDEX};
     % x_i <- center of R_i
     center = [regions{index, X_CENTER_INDEX}, regions{index, Y_CENTER_INDEX}];
     % sigma_i <- sqrt(var(p^u_xi))
     varianceablePatch = double(patch);
     sigma_i = sqrt(var(varianceablePatch(:)));
     
     % Compute y_i = arg_y min d[p^u_xi, p^v_y]
     [ds, newPatches] = computeArg(regions, index, style_image);
     minD = min(ds);
     minD = minD(1);
     newPatchIndex = find(ds==minD);
     y_i = newPatches(newPatchIndex);
     
     % if C(p^u_x, p^v_y) is true
     if ((sigma_i + minD > omega) && (width > min_width)) || (width > max_width)
         % Split R_i into four
         % R = {R \ R_i} U {R_m+1, ..., R_m+4}
         
         % make room for new rows
         if index < size(regions, 1)
            numToShift = size(regions, 1) - index - 1;
            regions(index+5:index+5+numToShift, :) = regions(index + 1:index+1+numToShift, :);
         end
         
         newPatch1 = base_image(regions{index,X_OFFSET_INDEX}:regions{index,X_OFFSET_INDEX}+width/2-1, regions{index,Y_OFFSET_INDEX}:regions{index,Y_OFFSET_INDEX}+width/2-1, :);
         newWidth = width/2;
         newXOffset1 = regions{index, X_OFFSET_INDEX};
         newYOffset1 = regions{index, Y_OFFSET_INDEX};
         newXCenter1 = newXOffset1 + newWidth/2;
         newYCenter1 = newYOffset1 + newWidth/2;
         regions{index+1, PATCH_INDEX} = newPatch1;
         regions{index+1, X_OFFSET_INDEX} = newXOffset1;
         regions{index+1, Y_OFFSET_INDEX} = newYOffset1;
         regions{index+1, X_CENTER_INDEX} = newXCenter1;
         regions{index+1, Y_CENTER_INDEX} = newYCenter1;
         regions{index+1, WIDTH_INDEX} = newWidth;
         
         newPatch2 = base_image(regions{index,X_OFFSET_INDEX}+width/2:regions{index,X_OFFSET_INDEX}+width-1, regions{index,Y_OFFSET_INDEX}:regions{index,Y_OFFSET_INDEX}+width/2-1, :);
         newXOffset2 = regions{index, X_OFFSET_INDEX} + width/2;
         newYOffset2 = regions{index, Y_OFFSET_INDEX};
         newXCenter2 = newXOffset2 + newWidth/2;
         newYCenter2 = newYOffset2 + newWidth/2;
         regions{index+2, PATCH_INDEX} = newPatch2;
         regions{index+2, X_OFFSET_INDEX} = newXOffset2;
         regions{index+2, Y_OFFSET_INDEX} = newYOffset2;
         regions{index+2, X_CENTER_INDEX} = newXCenter2;
         regions{index+2, Y_CENTER_INDEX} = newYCenter2;
         regions{index+2, WIDTH_INDEX} = newWidth;
         
         newPatch3 = base_image(regions{index,X_OFFSET_INDEX}:regions{index,X_OFFSET_INDEX}+width/2-1, regions{index,Y_OFFSET_INDEX}+width/2:regions{index,Y_OFFSET_INDEX}+width-1, :);
         newXOffset3 = regions{index, X_OFFSET_INDEX};
         newYOffset3 = regions{index, Y_OFFSET_INDEX} + width/2;
         newXCenter3 = newXOffset3 + newWidth/2;
         newYCenter3 = newYOffset3 + newWidth/2;
         regions{index+3, PATCH_INDEX} = newPatch3;
         regions{index+3, X_OFFSET_INDEX} = newXOffset3;
         regions{index+3, Y_OFFSET_INDEX} = newYOffset3;
         regions{index+3, X_CENTER_INDEX} = newXCenter3;
         regions{index+3, Y_CENTER_INDEX} = newYCenter3;
         regions{index+3, WIDTH_INDEX} = newWidth;
         
         newPatch4 = base_image(regions{index,X_OFFSET_INDEX}+width/2:regions{index,X_OFFSET_INDEX}+width-1, regions{index,Y_OFFSET_INDEX}+width/2:regions{index,Y_OFFSET_INDEX}+width-1, :);
         newXOffset4 = regions{index, X_OFFSET_INDEX} + width/2;
         newYOffset4 = regions{index, Y_OFFSET_INDEX} + width/2;
         newXCenter4 = newXOffset4 + newWidth/2;
         newYCenter4 = newYOffset4 + newWidth/2;
         regions{index+4, PATCH_INDEX} = newPatch4;
         regions{index+4, X_OFFSET_INDEX} = newXOffset4;
         regions{index+4, Y_OFFSET_INDEX} = newYOffset4;
         regions{index+4, X_CENTER_INDEX} = newXCenter4;
         regions{index+4, Y_CENTER_INDEX} = newYCenter4;
         regions{index+4, WIDTH_INDEX} = newWidth;
         
         % remove original patch that we split
         regions(index, :) = [];
         
     else
         % Compute spatially-constrained k-NN:
         % L_i <- {l_ik}^K_k=1 with |l_ik - l_ik+1| > X
         % labels are the centers of the patches
         % Compute k-NN over the patches of the style_image at different
         % centers
         % X = width/2
         candidatePatches = {};
         patchesIndex = 1;
         for i = 1:size(style_image, 1) - width
             for j = 1:size(style_image, 2) - width
                 newPatch = style_image(i:i+width-1, j:j+width-1, :);
                 candidatePatches{patchesIndex, 1} = newPatch;
                 candidatePatches{patchesIndex, 2} = i+width/2;
                 candidatePatches{patchesIndex, 3} = j+width/2;
                 patchesIndex = patchesIndex + 1;
             end
         end
         
         % need to include the spatial constraint
         neighborResults = computeKnn(patch, candidatePatches, K, width);
         candidateLabelIndex = size(candidateLabels, 1) + 1;
         candidateLabels{candidateLabelIndex, 1} = index;
         candidateLabels{candidateLabelIndex, 2} = neighborResults;
         
         
         
         if index < size(regions,1)
            index = index + 1;
         else
             done = true;
         end
     
     end
end

% at this point you have regions (R in paper) and candidateLabels (L in
% paper) to work with for the rest of the steps

%% step 2: optimization

% Use ICM to remove the noise from the given image.
% * covar is the known covariance of the Gaussian noise.
% * max_diff is the maximum contribution to the potential
%   of the difference between two neighbouring pixel values.
% * weight_diff is the weighting attached to the component of the potential
%   due to the difference between two neighbouring pixel values.
% * iterations is the number of iterations to perform.



% this janky code only works when K = 1. It entirely skips optimization
labels = {};
for i = 1:size(candidateLabels, 1)
    labels{i, 1} = regions(candidateLabels{i, 1}, :);
    candidateSet = candidateLabels{i, 2};
    % replace patch with original region to see if it looks normal
%     candidateSet{1, 1} = regions{candidateLabels{i, 1}, 1};
    labels{i, 2} = candidateSet(1, :);
end



% at end of this method, should have cell array called labels which
% contains in its first cell the cell array describing the region in the
% original image (includes all details from regions data structure - region
% contents, center, width, and offset, and in its second cell a cell array
% describing the patch in the style image (patch includes actual content of
% patch + center)
%% step 3: bilinear blending

image = stitchImage(labels);
figure, imshow(image);
imageBlur1 = applyEdgeBlend(image, labels);
figure, imshow(imageBlur1);
imageBlur2 = totalBlend(image, .7);
figure, imshow(imageBlur2);


%% step 4: global color and contrast matching

style_image = imread('w.jpg');
base_image = imread('kitten.jpg');

% chromatic adaptation transform

m_cat02 = [0.7328 0.4296 -0.1624; -0.7036 1.6975 0.0061; 0.0030 0.0136 0.9834];

% 1.Estimate the white point (illuminant) of image E. For a given value t (we
% set t = 0.3,  more  discussion  on  [9]),  the  white  point  of  an  
% image  is  defined as the mean color value of all pixels such that 
% (|aâˆ—|+|bâˆ—|)/Lâˆ— < t ,
% where aâˆ—, bâˆ— and Lâˆ— denote the pixel coordinates in the CIELAB color space.

% convert to other forms
lab_style = rgb2lab(style_image); 

% TODO: should be finished texture transfer image, not base image
lab_base = rgb2lab(base_image);

whitepoint_style = computeWhitepoint(lab_style);
xyz_wp_style = lab2xyz(whitepoint_style);
lms_wp_style = m_cat02*[xyz_wp_style(1,1);xyz_wp_style(1,2);xyz_wp_style(1,3)];

% 2.  Estimate similarly the white point of image I.
% 3.  Perform the chromatic adaptation transform (CAT) on I to adapt its 
% white point to the white point of E.
% 4.  Repeat Steps 2 and 3 until (a) the maximum number of iterations has 
% been reached or; (b) the I white point has not changed from the previous 
% iteration.
iter_num = 0;
max_iter = 30;
new_whitepoint = [0, 0, 0];
old_whitepoint = [0, 0, 0];
input_im = lab_base;

while iter_num < max_iter
   old_whitepoint = new_whitepoint; 
   
   % step 2
   new_whitepoint = computeWhitepoint(input_im);
   xyz_wp_new = lab2xyz(new_whitepoint);
   lms_wp_input = m_cat02*[xyz_wp_new(1,1);xyz_wp_new(1,2);xyz_wp_new(1,3)];
   
   % step 3 
   transform = inv(m_cat02)*diag([lms_wp_style(1,1)/lms_wp_input(1,1), ...
       lms_wp_style(2,1)/lms_wp_input(2,1), lms_wp_style(3,1)/lms_wp_input(3,1)])...
       *m_cat02;
   
   xyz_input = lab2xyz(input_im);
   new_im = zeros(size(xyz_input, 1), size(xyz_input,2), 3);
   
   for k = 1:size(xyz_input, 1)
       for l = 1:size(xyz_input, 2)
           new_pix = transform*[xyz_input(k,l,1);xyz_input(k,l,2);xyz_input(k,l,3)];
           new_im(k,l,1) = new_pix(1,1);
           new_im(k,l,2) = new_pix(2,1);
           new_im(k,l,3) = new_pix(3,1);
       end
   end
   
   input_im = xyz2lab(new_im);
   
   % end conditions
   if sum(new_whitepoint == old_whitepoint) == 3
       break;
   end
   iter_num = iter_num + 1; 
end    

% 5.  Return image Iâ€² which has the same geometry of I but with colors 
% adapted to the illuminant of E.

imshow(lab2uint8(input_im));

% color chroma transfer

% convert to hst
base_hsv_im = rgb2hsv(lab2rgb(input_im));
style_hsv_im = rgb2hsv(style_image);

% find signatures
base_sigs = hsvFTC(base_hsv_im);
style_sigs = hsvFTC(style_hsv_im);

sig_dist_matrix = [];

% create distance matrix
for i = 1:size(base_sigs, 1)
    sig_dist_matrix(i) = [];
    for j = 1:size(style_sigs, 1)
        sig_dist_matrix(i,j) = norm(base_sigs(i,1:3) - style_sigs(j,1:3)) + norm(base_sigs(i,4:6) - style_sigs(j,4:6));
    end
end    
   





%% ------------------- functions go below here -----------------------------

% used to calculate the minimum distance between a base patch and all
% possible style patches
function [ds, newPatches] = computeArg(regions, index, style_image)
    global WIDTH_INDEX;
    MY_WIDTH_INDEX = WIDTH_INDEX;
    global PATCH_INDEX;
    MY_PATCH_INDEX = PATCH_INDEX;
    patch = regions{index, MY_PATCH_INDEX};
    width = regions{index, MY_WIDTH_INDEX};
    ds = [];
    newPatches = [];
    for i = 1:size(style_image, 1)/width
        for j = 1:size(style_image, 2)/width
            newPatch = style_image(i:i+width-1, j:j+width-1, :);
            d = (abs(double(patch(:)) - double(newPatch(:))).^2)/(width^2);
            ds = [ds, d];
            newPatches = [newPatches, newPatch];
        end
    end
end

% used to check the spatial constraint when calculating k-NN
% function good = knnSpatialConstraint(center1X, center1Y, center2X, center2Y, patchWidth)
%     good = pdist([center1X, center1Y; center2X, center2Y], 'euclidean') > patchWidth/2;
% end

% used to compute k-NN
function neighbors = computeKnn(patch, candidatePatches, K, width)
    % patch, distance, centerX, centerY
    neighbors = {};
    for i = 1:size(candidatePatches)
        distance = sqrt(sum((patch(:) - candidatePatches{i, 1}(:)) .^ 2));
        better = false;
        worstIndex = -1;
        for j = 1:size(neighbors, 1)
            if neighbors{j, 2} > distance
                better = true;
                if worstIndex == -1
                    worstIndex = j;
                elseif neighbors{worstIndex, 2} < neighbors{j, 2}
                    worstIndex = j;
                end
            end
        end
        
        
        if better == true || size(neighbors, 1) < K
            newNeighbors = neighbors(:,:);
            if worstIndex >= 1
                newNeighbors{worstIndex, 1} = candidatePatches{i, 1};
                newNeighbors{worstIndex, 2} = distance;
                newNeighbors{worstIndex, 3} = candidatePatches{i, 2};
                newNeighbors{worstIndex, 4} = candidatePatches{i, 3};
            else
                neighborsIndex = size(newNeighbors, 1) + 1;
                newNeighbors{neighborsIndex, 1} = candidatePatches{i, 1};
                newNeighbors{neighborsIndex, 2} = distance;
                newNeighbors{neighborsIndex, 3} = candidatePatches{i, 2};
                newNeighbors{neighborsIndex, 4} = candidatePatches{i, 3};
            end
            % should replace unless spatial constraint prohibits
            failedConstraint = false;
            for j = 1:size(neighbors, 1)
                if sqrt((candidatePatches{i, 2} - neighbors{j, 3}).^2 + (candidatePatches{i, 3} - neighbors{j, 4}).^2) > width/2 == false
                    failedConstraint = true;
                end
            end
            if failedConstraint == false
                neighbors = newNeighbors(:,:);
            end
        end
    end
end
% comptes the whitepoint of an input image given in CIELAB form
function whitepoint = computeWhitepoint(input_image)
    white_pixels = [];
    t = 0.3;
    for k = 1:size(input_image, 1)
        for l = 1:size(input_image, 2)
           if (abs(input_image(k, l, 1)) + abs(input_image(k, l, 2)))/input_image(k, l, 3) < t
               white_pixels = [white_pixels; input_image(k, l, :)];
           end
        end
    end    
    whitepoint = [sum(white_pixels(:, 1, 1))/length(white_pixels), ...
        sum(white_pixels(:, 1, 2))/length(white_pixels), ...
        sum(white_pixels(:, 1, 3))/length(white_pixels)]; 
end

% used to see if a gap is meaningful, takes histogram and lower and upper bound of segment
function meaningful = isMeaningful(h, lower_bound, upper_bound)
    a = find(h.BinEdges == lower_bound);
    b = find(h.BindEdges == upper_bound);
    N = size(h.Data);
    L = h.NumBins;
    epsilon = 1;
    meaningful_threshold = (1/N)*log10((L*(L+1))/(2*epsilon));
    r = 0;
    for bin = a:b-1
        r = r + h.BinCount(bin);    
    end
    p = (b-a+1)/L;
    H = r*log10(r/p)+(1-r)*log((1-r)/(1-p));
    meaningful = H > meaningful_threshold;
end

function S = fineToCoarseSegmentation(r)
    % calculate r_bar - Pool Adjacent Violators Algorithm
    %r_bar = histogram(r.Data, unique(r.Data));
    edge = 2;
    map_to_new_bins = [1];
    n = 1;
    new_bin_edges = [r.BinEdges(1)];
    new_bin_count = [r.BinCounts(1)];

    for m = 2:size(r.BinCounts)
        n = n+1;
        new_bin_edges(n) = r.BinEdges(m);
        new_bin_count(n) = r.BinCounts(m);
        while (n > 1 && new_bin_count(n) < new_bin_count(n-1))
            new_bin_edges(n) = [];
            new_bin_count(n-1) = new_bin_count(n-1) + new_bin_count(n);
            map_to_new_bins(m) = n;
            n = n - 1;
        end
    end

    %r_bar.BinEdges = [new_bin_edges, r.BinEdges(end)];
    %r_bar.BinCount = new_bin_count;       

    % step 1 of FTC: find local minima
    [loc_mins, min_indices] = findpeaks(-1*(r.Data));
    % Initialize S={s0,…,sn} as the finest segmentation of the histogram, i.e., the list of all the local minima, plus the endpoints s0=1 and sn=L.
    S = [0 + min_indices + 1];

    % step 2 and 3 of FTC

    keep_going = true;
    num_iter_while = 0;
    indices_checked = [];

    for o = 2:size(S)
        while(keep_going)
            % Choose i randomly in [1,size(S)−1] (adjusted for 1-indexing to be [2,size(S)-(o-1)])
            i = 2+rand*(size(S)-(o+1));
            % If the modes on both sides Formula can be gathered in a single interval Formula following the unimodal hypothesis, group them. Update Formula.
            % find c between s-1 and s+o
            c = 0;
            for k = S(i-1)+r.BinWidth:r.BinWidth:S(i+o)-r.BinWidth
                r_ac = sum(r.BinCounts(i-1:k));
                r_bar_ac = sum(r.BinCounts(S(i-1):S(k)));
                if ((r_ac >= r_bar_ac) && (not (isMeaningful(r, i-1, k))))
                    r_cb = sum(r.BinCounts(k:i+o-1));
                    r_bar_cb = sum(r.BinCounts(S(k):S(i+o-1)));
                    if ((r_cb <= r_bar_cb) && (not (isMeaningful(r, k, i+1))))
                        c = k;
                        % remove all sections beyond i-1
                        for p = 1:o
                            S(i) = [];
                        end
                        % remove all sections absorbed into new section and adjacent sections from indices_checked
                        if ismember(i-2, indices_checked)
                            indices_checked(i-2) = [];
                        end
                        if ismember(i-1, indices_checked)
                            indices_checked(i-1) = [];
                        end
                        if ismember(i, indices_checked)
                            indices_checked(i) = [];
                        end
                        for q = 1:o
                            if ismember(i+q, indices_checked)
                                indices_checked(i+q) = [];
                            end
                        end
                        % adjust indices of all remaining entries in indices_checked to account for o merged sections
                        for l = 1:size(indices_checked)
                            if indices_checked(l) > i
                                indices_checked(l) = indices_checked(l)-o;
                            end
                        end
                        break;
                    end
                end
            end
            if c == 0
                indices_checked = [indices_checked, i];
            end
            if size(indices_checked) == (size(S)-o)
                keep_going = false;
            end
        end             
    end
end

% takes an image (in HSV color form) and returns the modes returned by the fine to coarse segmentation algorithm in the form of signatures, where each mode is represented by a vector of the means and standard deviations of the cielab form of the mode
function signatures = hsvFTC(im)
    % select hues for histogram
    hue_histogram_points = [];

    for i = 1:size(im, 1)
        for j = 1:size(im, 2)
           hue_histogram_points = [hue_histogram_points, im(i, j, 1)];
        end
    end
    hue_hist = histogram(hue_histogram_points, unique(hue_histogram_points));   
    
    hue_segments = fineToCoarseSegmentation(hue_hist);
        
    loop_hue_hist_points = im;
    loop_sat_hist_points = im;
    master_color_points = im;

    signatures = [];

    % repeat FTC for saturation
    for s = 1:(size(hue_segments,2) - 1)
        sat_hist_points = [];
        for t = 1:size(loop_hue_hist_points, 1)
            for u = 1:size(loop_hue_hist_points, 2)
                if loop_hue_hist_points(t, u, 1) < hue_segments(1,s+1)
                    sat_hist_points = [sat_hist_points, loop_hue_hist_points(t, u, 2)];
                end
            end
        end
        sat_hist = histogram(sat_hist_points, unique(sat_hist_points));
        sat_segments = fineToCoarseSegmentation(sat_hist);

        % repeat FTC for value
        for v = 1:(size(sat_segments,2) - 1)
            val_hist_points = [];
            for w = 1:size(loop_sat_hist_points, 1)
                for x = 1:size(loop_sat_hist_points, 2)
                    if loop_sat_hist_points(w, x, 1) < hue_segments(1,s+1) && loop_sat_hist_points(w, x, 2) < sat_segments(1,v+1)
                        val_hist_points = [val_hist_points, loop_sat_hist_points(w, x, 3)];
                    end
                end
            end
            val_hist = histogram(val_hist_points, unique(val_hist_points));
            val_segments = fineToCoarseSegmentation(val_hist);

            % find representative color modes and make signatures
            for y = 1:(size(val_segments,2) - 1)
                l_list = [];
                a_list = [];
                b_list = [];
                for z = 1:size(master_color_points, 1)
                    for aa = 1:size(master_color_points, 2)
                        if  master_color_points(z, aa, 1) < hue_segments(1,s+1) && master_color_points(z, aa, 2) < sat_segments(1,v+1) && master_color_points(z, aa, 3) < val_segments(1,y+1)
                            lab_color = rgb2lab(hsv2rgb([master_color_points(z, aa, 1), master_color_points(z, aa, 2), master_color_points(z, aa, 3)]));
                            l_list = [l_list, lab_color(1)];
                            a_list = [a_list, lab_color(2)];
                            b_list = [b_list, lab_color(3)];
                        end
                    end
                end
                signatures = [signatures; [mean(l_list), mean(a_list), mean(b_list), std(l_list), std(a_list), std(b_list)]];
            end
        end
    end
end

function dst = restore_image(src, covar, max_diff, weight_diff, iterations)

    % Maintain two buffer images.
    % In alternate iterations, one will be the
    % source image, the other the destination.
    buffer = zeros(size(src,1), size(src,2), 2);
    buffer(:,:,1) = src;
    s = 2;
    d = 1;

    % This value is guaranteed to be larger than the
    % potential of any configuration of pixel values.
    V_max = (size(src,1) * size(src,2)) * ...
            ((256)^2 / (2*covar) + 4 * weight_diff * max_diff);

    for i = 1 : iterations

        % Switch source and destination buffers.
        if s == 1
            s = 2;
            d = 1;
        else
            s = 1;
            d = 2;
        end

        % Vary each pixel individually to find the
        % values that minimise the local potentials.
        for r = 1 : size(src,1)
            for c = 1 : size(src,2)

                V_local = V_max;
                min_val = -1;
                for val = 0 : 255

                    % The component of the potential due to the known data.
                    V_data = (val - src(r,c))^2 / (2 * covar);

                    % The component of the potential due to the
                    % difference between neighbouring pixel values.
                    V_diff = 0;
                    if r > 1
                        V_diff = V_diff + ...
                        min( (val - buffer(r-1,c,s))^2, max_diff );
                    end
                    if r < size(src,1)
                        V_diff = V_diff + ...
                        min( (val - buffer(r+1,c,s))^2, max_diff );
                    end
                    if c > 1
                        V_diff = V_diff + ...
                        min( (val - buffer(r,c-1,s))^2, max_diff );
                    end
                    if c < size(src,2)
                        V_diff = V_diff + ...
                        min( (val - buffer(r,c+1,s))^2, max_diff );
                    end

                    V_current = V_data + weight_diff * V_diff;

                    if V_current < V_local
                        min_val = val;
                        V_local = V_current;
                    end

                end

                buffer(r,c,d) = min_val;
            end
        end

        i
    end

    dst = buffer(:,:,d);
    
end

function image = stitchImage(labels)
    % cell 1 of labels contains cell array for region of base image
    % cell 2 of labels contains cell array of patch for new image
    global totalWidth;
    global X_OFFSET_INDEX;
    global Y_OFFSET_INDEX;
    global WIDTH_INDEX;
    
    image = zeros(totalWidth, totalWidth, 3, 'uint8');
    for i = 1:size(labels)
       region = labels{i, 1};
       patch = labels{i, 2};
       xOff = region{X_OFFSET_INDEX};
       yOff = region{Y_OFFSET_INDEX};
       patchWidth = region{WIDTH_INDEX};
       applyPatch = patch{1};
       for j = 1:size(applyPatch, 1)
           for k = 1:size(applyPatch, 2)
               image(xOff+j-1, yOff+k-1, 1) = applyPatch(j, k, 1);
               image(xOff+j-1, yOff+k-1, 2) = applyPatch(j, k, 2);
               image(xOff+j-1, yOff+k-1, 3) = applyPatch(j, k, 3);
           end
       end
%        image(xOff:xOff+patchWidth-1, yOff:yOff+patchWidth-1, 1) = applyPatch(:, :, 1);
%        image(xOff:xOff+patchWidth-1, yOff:yOff+patchWidth-1, 2) = applyPatch(:, :, 2);
%        image(xOff:xOff+patchWidth-1, yOff:yOff+patchWidth-1, 3) = applyPatch(:, :, 3);
    end
    
end

function image = applyEdgeBlend(image, labels)
    % cell 1 of labels contains cell array for region of base image
    % cell 2 of labels contains cell array of patch for new image
    global totalWidth;
    global X_OFFSET_INDEX;
    global Y_OFFSET_INDEX;
    global WIDTH_INDEX;
    
    for i = 1:size(labels)
       region = labels{i, 1};
       
       xOffset = region{X_OFFSET_INDEX};
       yOffset = region{Y_OFFSET_INDEX};
       width = region{WIDTH_INDEX};
       
       % if there's something to blur with to the left and right
       if xOffset > 2 && (totalWidth - xOffset - width) > 2 && (yOffset + width) <= totalWidth && yOffset > 1
           x1 = xOffset - 2;
           y1 = yOffset;
           x2 = xOffset + 2;
           y2 = yOffset + width;
           
           
           blurredRed = uint8(conv2(double(image(:, :, 1)), [0,1,0;1,0,1;0,1,0]/4, 'same'));
           blurredGreen = uint8(conv2(double(image(:, :, 2)), [0,1,0;1,0,1;0,1,0]/4, 'same'));
           blurredBlue = uint8(conv2(double(image(:, :, 3)), [0,1,0;1,0,1;0,1,0]/4, 'same'));
           
           blurred = cat(3, blurredRed, blurredGreen, blurredBlue);
           image(x1:x2, y1:y2) = blurred(x1:x2, y1:y2);
       end
       
       % if there's something to blur with above and below
       if yOffset > 2 && (totalWidth - yOffset) > 2 && (xOffset + width) <= totalWidth && xOffset > 1
           x1 = xOffset;
           y1 = yOffset - 2;
           x2 = xOffset + width;
           y2 = yOffset + 2;
           
           
           blurredRed = uint8(conv2(double(image(:, :, 1)), [0,1,0;1,0,1;0,1,0]/4, 'same'));
           blurredGreen = uint8(conv2(double(image(:, :, 2)), [0,1,0;1,0,1;0,1,0]/4, 'same'));
           blurredBlue = uint8(conv2(double(image(:, :, 3)), [0,1,0;1,0,1;0,1,0]/4, 'same'));
           
           blurred = cat(3, blurredRed, blurredGreen, blurredBlue);
           image(x1:x2, y1:y2, 1) = blurred(x1:x2, y1:y2, 1);
           image(x1:x2, y1:y2, 2) = blurred(x1:x2, y1:y2, 2);
           image(x1:x2, y1:y2, 3) = blurred(x1:x2, y1:y2, 3);
       end
    end
    
end

function image = totalBlend(image, howBlur)
    image = imgaussfilt(image, howBlur);
end



