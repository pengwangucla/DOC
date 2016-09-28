addpath('./lib')
cmap = VOClabelcolormap();

dataset='PIOD';
set_path
%% set up 
[occ_model, edge_model, opt] = get_config_info();

%%
edge_path = [ResPath,  edge_model, '/']; 
occ_path = [ResPath,  occ_model, '/']; 

for iimg = 1:length(image_list)
    fprintf([image_list{iimg}, '\n']);
    im = imread([ImgPath, image_list{iimg}, '.jpg']);
    occ_file = [occ_path, image_list{iimg}, '.mat'];
    edge_file = [edge_path, image_list{iimg}, '.mat'];
    doc_res = im2occedge(occ_file, edge_file, opt);
    
    % load ground truth
    
    load([GTPath, image_list{iimg}, '.mat'], 'bndinfo_pascal');
    gt_img = GetDilateEdgeGT(bndinfo_pascal);

    s_num = 1;
    r= 3;
    c = 2;
    subplot_tight(r, c, 1); imshow(im);
    subplot_tight(r,c,3); imshow(doc_res{1,1},[]);
    subplot_tight(r,c,4); imagesc(doc_res{1,2});
    colormap(parula(100));  axis('image');
    colorbar('FontSize',12);
    freezeColors;
    
    subplot_tight(r,c,5); imshow(gt_img(:,:,1));
    subplot_tight(r,c,6); imagesc(gt_img(:,:,2));  axis('image');
    colormap(parula(100));
    colorbar('FontSize',12);
    freezeColors;
    
    pause;
    close all;

end
