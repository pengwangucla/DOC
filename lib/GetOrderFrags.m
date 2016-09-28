function [frags, new_edge] = GetOrderFrags(thin_edge, len_thresh)
if ~exist('len_thresh','var'); len_thresh = 5; end 
new_edge = zeros(size(thin_edge)); 
max_range = 8; 
imsize = size(thin_edge); 
CC = bwconncomp(thin_edge,max_range); 
frags = [];
for icc = 1:length(CC.PixelIdxList)
    if length(CC.PixelIdxList{icc}) < len_thresh; continue; end 
    frags = [frags, OrderByConnect(CC.PixelIdxList{icc}, imsize )]; 
end

for ifrag = 1:length(frags)
    new_edge(frags{ifrag}) = 1; 
end

end