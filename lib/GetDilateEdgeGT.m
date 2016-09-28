function sem_map = GetDilateEdgeGT(bndinfo, varargin)
opt = struct('dilate', 1);
opt = CatVarargin(opt, varargin);

se = strel('disk', opt.dilate);

if bndinfo.ne == 0
    sem_map = zeros([bndinfo.imsize, 2], 'single');
    
else
    % generate prediction ground truth map
    sem_map = zeros(bndinfo.imsize, 'single');
    edgemap = zeros(bndinfo.imsize, 'single');
    for iedge = 1:length(bndinfo.edges.indices);
        edgemap(bndinfo.edges.indices{iedge}) = 1;
    end
    
    if opt.dilate % whether dilate the ground truth
        sem_map(:,:,1) = imdilate(edgemap, se);
        
        % for orientation map
        temp = bndinfo.OrientMap;
        temp(temp ~= 0) = temp( temp ~=0) + 2*pi;
        temp = imdilate(temp, se);
        temp(temp ~= 0) = temp( temp ~= 0) - 2*pi;
        sem_map(:,:,2) = temp;
    else
        sem_map(:,:,1) = edgemap;
        sem_map(:,:,2) = bndinfo.OrientMap;
    end
    %... can add other ground truth here
end

end