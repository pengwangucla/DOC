function frags = OrderByConnect(ind, imsize); 

[y, x] = ind2sub(imsize, ind); 
x = single(x); 
y = single(y); 
x_dis = single(abs(bsxfun(@minus, x(:), x(:)'))); 
y_dis = single(abs(bsxfun(@minus, y(:), y(:)'))); 
x_dis = setdiag(x_dis,inf); 
y_dis = setdiag(y_dis,inf); 

dis_mat = single(cat(3, x_dis, y_dis)); 
adjmat = max(dis_mat, [], 3) <= 1; 
dis_mat = sum(dis_mat, 3); 
frags = PropAdj(adjmat, dis_mat); 
 
for ifrag = 1:length(frags)
    frags{ifrag} = ind(frags{ifrag}); 
end
end 

function groups = PropAdj(adj, dismat)
    
np = size(adj,1); 

set_flag = false(np, 1); 
groups = cell(1); 
len = ones(1); 
igrp = 1; 
for ipx = 1:np
    if set_flag(ipx) 
        continue; 
    end
    cur_id = ipx; 
    ind = [ipx]; 
    set_flag(ipx) = 1; 
    len(igrp) =1 ; 
    while 1
        id_next = find(adj(cur_id, :)); 
        if numel(id_next) == 0
            break; 
        end
        if numel(id_next) > 1; 
                [~, idx] = min(dismat(cur_id, id_next)); id_next = id_next(idx); 
        end
        % can not back to self
        adj(cur_id,:) = 0; adj(:, cur_id) = 0; 
        ind = [ind, id_next]; 
        set_flag(id_next) = 1; 
        len(igrp) = len(igrp) + 1; 
        cur_id = id_next; 
    end
    groups{igrp} = ind ; 
    igrp = igrp + 1; 
end

% rm the isolated one 
groups( len <= 2) = []; 
end
