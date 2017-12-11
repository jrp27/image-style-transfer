% requires base_image to be a square and style_image to be equal to or
% larger in size than base_image along both dimensions
style_image = imread('pointillism.jpg');
base_image = imread('cat.jpg');

% step 1: split and match
min_width = 8;
max_width = 256;
omega = 15;
K = 4;

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
             if index + 4 < size(regions, 1)
                numToShift = size(regions, 1) - index - 1;
                regions(index+5:index+5+numToShift, :) = regions(index + 1:index+1+numToShift, :);
             else
                 numToShift = 2;
                 regions(index+5:index+5+numToShift, :) = regions(index+1:index+1+numToShift, :);
             end
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

%% step 3: bilinear blending

%% step 4: global color and contrast matching

style_image = imread('pontillism.jpg');
base_image = imread('cat.jpg');

% chromatic adaptation transform

m_cat02 = [0.7328 0.4296 -0.1624; -0.7036 1.6975 0.0061; 0.0030 0.0136 0.9834];

% 1.Estimate the white point (illuminant) of image E. For a given value t (we
% set t = 0.3,  more  discussion  on  [9]),  the  white  point  of  an  
% image  is  defined as the mean color value of all pixels such that 
% (|a∗|+|b∗|)/L∗ < t ,
% where a∗, b∗ and L∗ denote the pixel coordinates in the CIELAB color space.

% convert to other forms
lab_style = rgb2lab(style_image); 

% TODO: should be finished texture transfer image, not base image
lab_base = rgb2lab(base_image);

whitepoint_style = computeWhitepoint(lab_style);

lms_wp_style = m_cat02*lab2xyz(whitepoint_style);

% 2.  Estimate similarly the white point of image I.
% 3.  Perform the chromatic adaptation transform (CAT) on I to adapt its 
% white point to the white point of E.
% 4.  Repeat Steps 2 and 3 until (a) the maximum number of iterations has 
% been reached or; (b) the I white point has not changed from the previous 
% iteration.
itern_num = 0;
max_iter = 30;
new_whitepoint = [];
old_whitepoint = [];
input_im = lab_base;

while iternum < max_iter
   old_whitepoint = new_whitepoint; 
   
   % step 2
   new_whitepoint = computeWhitepoint(input_im);
   
   lms_wp_input = m_cat02*lab2xyz(new_whitepoint);
   
   % step 3 
   transform = inv(m_cat02)*diag([lms_wp_style(1)/lms_wp_input(1), ...
       lms_wp_style(2)/lms_wp_input(2), lms_wp_style(3)/lms_wp_input(3)])...
       *m_cat02;
   
   xyz_input = lab2xyz(input_im);
   new_im = [];
   
   for k = 1:length(xyz_input)
       new_im = [new_im; transform*xyz_input(k)];
   end
   
   input_im = xyz2lab(new_im);
   
   % end conditions
   if sum(new_whitepoint == old_whitepoint) == 3
       break;
   end
   iternum = iternum + 1; 
end    

% 5.  Return image I′ which has the same geometry of I but with colors 
% adapted to the illuminant of E.

imshow(input_im);

% color chroma transfer






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
    disp(input_image(1, 1,:));
    for k = 1:size(input_image, 1)
        for l = 1:size(input_image, 2)
           if (abs(input_image(k, l, 1)) + abs(input_image(k, l, 2)))/input_image(k, l, 3) < t
               % TODO: fix adding to matrix
               white_pixels = [white_pixels; input_image(k, l, :)];
           end
        end
    end    
    disp(size(white_pixels));
    whitepoint = [sum(white_pixels, 1)/length(white_pixels), ...
        sum(white_pixels, 2)/length(white_pixels), ...
        sum(white_pixels, 3)/length(white_pixels)]; 
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
