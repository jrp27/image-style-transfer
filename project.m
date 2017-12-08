style_image = imread("pontillism.jpg");
base_image = imread("cat.jpg");

% step 1: split and match
min_width = 8^2;
max_width = 256^2;
omega = 15;
K = 4;

patch1 = base_image;
patches = [patch1];
offsets = [1, 1];
centers = [];
widths = [size(base_image, 1)];
candidateLabels = [];

index = 1;
done = false;

while not(done)
    % R_i
     patch = patches(index);
     width = widths(index);
     % x_i <- center of R_i
     center = size(patch, 1)/2 + offsets(index, 1), size(patch, 2)/2 + offsets(index, 2);
     centers = [centers, center];
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
         newPatch1 = base_image(offset(index,1):offset(index,1)+width/2, offset(index,2):offset(index,2)+width/2, :);
         newPatch2 = base_image(offset(index,1)+width/2:offset(index,1)+width, offset(index,2):offset(index,2)+width/2, :);
         newPatch3 = base_image(offset(index,1):offset(index,1)+width/2, offset(index,2)+width/2:offset(index,2)+width, :);
         newPatch4 = base_image(offset(index,1)+width/2:offset(index,1)+width, offset(index,2)+width/2:offset(index,2)+width, :);
         
         % R = {R \ R_i} U {R_m+1, ..., R_m+4}
         patches = patches(patches~=patch);
         patches = [patches, newPatch1, newPatch2, newPatch3, newPatch4];
         % TODO SOMEHOW HANDLE WIDTHS HERE
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
         
         % actually need to write own k-NN to be able to include the
         % spatial constraint!
         [newLabels, distances] = knnsearch([patch], candidatePatches, 'K', K);
         
         
         if index < size(patches,1)
            index = index + 1;
         else
             done = true;
     end
     
end

% at this point you have patches (R in paper) and candidateLabels (L in
% paper) to work with for the rest of the steps

% step 2: optimization

% step 3: bilinear blending

% step 4: global color and contrast matching






% ------------------- functions go below here -----------------------------

% used to calculate the minimum distance between a base patch and all
% possible style patches
function ds, newPatches = computeArg(patch, index, style_image)
    width = widths(index);
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
    good = abs(center1 - center2) > width/2
end
     
     
    

