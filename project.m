style_image = imread('pointillism.jpg');
base_image = imread('cat.jpg');

% step 1: split and match
min_width = 8^2;
max_width = 256^2;
omega = 15;
K = 4;

% regions is: patch, xOffset, yOffset, xCenter, yCenter, width;
regions = [base_image, 0, 0, size(base_image, 1)/2, size(base_image, 2)/2, size(base_image,1)];
PATCH_INDEX = 1;
X_OFFSET_INDEX = 2;
Y_OFFSET_INDEX = 3;
X_CENTER_INDEX = 4;
Y_CENTER_INDEX = 5;
WIDTH_INDEX = 6;

candidateLabels = [];

index = 1;
done = false;

while not(done)
    % R_i
     patch = regions(index, PATCH_INDEX);
     width = regions(index, WIDTH_INDEX);
     % x_i <- center of R_i
     center = [regions(index, X_CENTER_INDEX), regions(index, Y_CENTER_INDEX)];
     % sigma_i <- sqrt(var(p^u_xi))
     sigma_i = sqrt(var(patch));
     
     % Compute y_i = arg_y min d[p^u_xi, p^v_y]
     ds, newPatches = computeArg(patch, index);
     minD = min(ds);
     newPatchIndex = find(ds==minD);
     y_i = newPatches(newPatchIndex);
     
     % if C(p^u_x, p^v_y) is true
     if (sigma_i + minD > omega && width > min_width) || width > max_width
         newWidth = width/2;
         % Split R_i into four
         % R = {R \ R_i} U {R_m+1, ..., R_m+4}
         removeCondition = regions(:, 1)==patch;
         regions(removeCondition) = [];
         
         newPatch1 = base_image(regions(index,X_OFFSET_INDEX):regions(index,X_OFFSET_INDEX)+width/2, regions(index,Y_OFFSET_INDEX):regions(index,Y_OFFSET_INDEX)+width/2, :);
         newWidth = width/2;
         newXOffset1 = regions(index, X_OFFSET_INDEX);
         newYOffset1 = regions(index, Y_OFFSET_INDEX);
         newXCenter1 = newXOffset1 + newWidth/2;
         newYCenter1 = newYOffset1 + newWidth/2;
         newRegion1 = [newPatch1, newXOffset1, newYOffset1, newXCenter1, newYCenter1, newWidth];
         
         newPatch2 = base_image(regions(index,X_OFFSET_INDEX)+width/2:regions(index,X_OFFSET_INDEX)+width, regions(index,Y_OFFSET_INDEX):regions(index,Y_OFFSET_IMAGE)+width/2, :);
         newXOffset2 = regions(index, X_OFFSET_INDEX) + width/2;
         newYOffset2 = regions(index, Y_OFFSET_INDEX);
         newXCenter2 = newXOffset2 + newWidth/2;
         newYCenter2 = newYOffset2 + newWidth/2;
         newRegion1 = [newPatch2, newXOffset2, newYOffset2, newXCenter2, newYCenter2, newWidth];
         
         newPatch3 = base_image(regions(index,X_OFFSET_INDEX):regions(index,X_OFFSET_INDEX)+width/2, regions(index,Y_OFFSET_INDEX)+width/2:regions(index,Y_OFFSET_INDEX)+width, :);
         newXOffset3 = regions(index, X_OFFSET_INDEX);
         newYOffset3 = regions(index, Y_OFFSET_INDEX) + width/2;
         newXCenter3 = newXOffset3 + newWidth/2;
         newYCenter3 = newYOffset3 + newWidth/2;
         newRegion3 = [newPatch3, newXOffset3, newYOffset3, newXCenter3, newYCenter3, newWidth];
         
         newPatch4 = base_image(regions(index,X_OFFSET_INDEX)+width/2:regions(index,X_OFFSET_INDEX)+width, regions(index,Y_OFFSET_INDEX)+width/2:regions(index,Y_OFFSET_INDEX)+width, :);
         newXOffset4 = regions(index, X_OFFSET_INDEX) + width/2;
         newYOffset4 = regions(index, Y_OFFSET_INDEX) + width/2;
         newXCenter4 = newXOffset4 + newWidth/2;
         newYCenter4 = newYOffset4 + newWidth/2;
         newRegion4 = [newPatch4, newXOffset4, newYOffset4, newXCenter4, newYCenter4, newWidth];
         
         regions = [regions; newRegion1; newRegion2; newRegion3; newRegion4];
     else
         % Compute spatially-constrained k-NN:
         % L_i <- {l_ik}^K_k=1 with |l_ik - l_ik+1| > X
         % labels are the centers of the patches
         % Compute k-NN over the patches of the style_image at different
         % centers
         % X = width/2
         candidatePatches = [];
         for i = 1:size(style_image, 1)/width
             for j = 1:size(style_image, 2)/width
                 newPatch = style_image(i:i+width, j:j+width, :);
                 candidatePatches = [candidatePatches, newPatch];
             end
         end
         
         % need to include the spatial constraint
         [newLabels, distances] = computeKnn(patch, candidatePatches, K);
         candidateLabels = [candidateLabels; newLabels];
         
         
         
         if index < size(patches,1)
            index = index + 1;
         else
             done = true;
        end
     
     end
end

% at this point you have patches (R in paper) and candidateLabels (L in
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

style_image = imread('pointillism.jpg');
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

%convert to hst
hsv_im = rgb2hsv(lab2rgb(input_im));

%select hues for histogram
histogram_points = [];

for i = 1:size(hsv_im, 1)
    for j = 1:size(hsv_im, 2)
       histogram_points = [histogram_points, hsv_im(i, j, 1);
    end
end

h = histogram(histogram_points);            
            






%% ------------------- functions go below here -----------------------------

% used to calculate the minimum distance between a base patch and all
% possible style patches
function [ds, newPatches] = computeArg(patch, index, style_image)
    width = regions(index, WIDTH_INDEX);
    ds = [];
    newPatches = [];
    for i = 1:size(style_image, 1)/width
        for j = 1:size(style_image, 2)/width
            newPatch = style_image(i:i+width, j:j+width, :);
            d = (abs(patch - newPatch).^2)/width^2;
            ds = [ds, d];
            newPatches = [newPatches, newPatch];
        end
    end
end

% used to check the spatial constraint when calculating k-NN
function good = knnSpatialConstraint(center1, center2, width)
    good = abs(center1 - center2) > width/2;
end

% used to compute k-NN
function [outLabels, distances] = computeKnn(patch, candidatePatches, K, width)
    [outLabels, distances] = knnsearch([patch], candidatePatches, 'K', K, 'Distance', 'euclidean');
    allGood = true;
    newCandidates = candidatePatches;
    for i = 1:size(outLabels)-1
        good = knnSpatialConstraint(outLabels(i), outLabels(i+1), width);
        if not(good)
            allGood = false;
            newCandidates = newCandidates(newCandidates~=candidatePatches(i));
            break;
        end
    end
    if not(allGood)
        computeKnn(patch, newCandidates, K, width);
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
