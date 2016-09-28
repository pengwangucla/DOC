function [occ_model, edge_model, config] =  get_config_info()

occ_model  = 'doc_ori';
edge_model = 'doc_edge';
config.thresh = 0.3;
% one may extend to multi scale for occlusion inference which we 
% do not include in this version.
config.occ_scale = 1;

end