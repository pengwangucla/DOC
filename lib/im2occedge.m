function doc_res = im2occedge(occ_file, edge_file, varargin)

model = struct('method', 'hed', 'resize_method', 'warp', ...,
    'scales', 1, 'thresh', 0.7, 'correct_theta', 1, 'occ_normalized',...,
    0, 'vis', 0, 'imsize', [], 'o_name', 'occmap', 'e_name', 'edgemap');
model = CatVarargin(model, varargin);

docmaps = cell(length(model.scales),2);
load(edge_file, 'edgemap');
docmaps{1,1} = edgemap;  % the first

load(occ_file, 'occmap');
if ~all(size(occmap) - size(docmaps{1,1}) == 0);
    occmap = imresize(occmap, size(docmaps{1,1}));
end
docmaps{1,2} = occmap;
docmaps{1,2} = max(min(docmaps{1,2}, 3*pi/2), -3*pi/2);

doc_res = cell(size(docmaps,1), 4);

[doc_res{1,4},doc_res{1, 2}, doc_res{1, 1}, doc_res{1,3}]  = ...,
    post_process(docmaps{1, 1}, docmaps{1, 2}, ...,
    model.thresh);

end


function [new_edge, new_occ, varargout] = post_process(edge_pred, ...,
    edge_occ, thresh)

assert(all(size(edge_pred(:,:,1)) - size(edge_occ(:,:,1)) == 0));
thin_edge = edge_nms(edge_pred, thresh);

new_occ = zeros(size(thin_edge), 'single');
sim_score = zeros(size(thin_edge), 'single');

[frags, new_edge] = GetOrderFrags(thin_edge, 5);
new_edge = single(new_edge);

for ifrag = 1:length(frags)
    [theta, s_score] = getTheta(frags{ifrag}, edge_occ);
    sim_score(frags{ifrag}) = s_score;
    new_occ(frags{ifrag}) = theta;
end

if nargout > 2
    varargout{1} = thin_edge;
    varargout{2} = sim_score; % whether the edge prediction and orientation prediction is similar
end
end

function [theta, varargout] = getTheta(pixel_id, edge_occ)

imsize = size(edge_occ); 
neighbor = 5; 

npix = length(pixel_id); 
idx = 1:npix; 
[y1, x1] = ind2sub(imsize, pixel_id(min(idx+neighbor, npix))); 
[y2, x2] = ind2sub(imsize(1:2), pixel_id(max(idx-neighbor,1)));

theta_1 = atan2(y2-y1, x2-x1); 
theta_2 = atan2(y1-y2, x1-x2); 

theta_pred = edge_occ(pixel_id);
abs_diff = abs(theta_pred-theta_1); abs_diff = mod(abs_diff, 2*pi);
ind_same = abs_diff  <= pi/2 |  abs_diff  > 3*pi/2;
theta = ind_same .* theta_1 + ~ind_same .* theta_2 ;

if nargout > 1
    varargout{1} = abs(cos(abs_diff)); 
end

end
