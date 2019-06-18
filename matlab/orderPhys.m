clear all, close all, clc;

% Rearrange indicators corresponding to the same audio
load rep_glb;
load rep_src;

[~, iname_glb] = sort(glb.name);
phys.name = glb.name(iname_glb, :);
phys.type = glb.type(iname_glb, :);
phys.playbackLeq = glb.playbackLeq(iname_glb, :);
phys.LAeq = glb.LAeq(iname_glb, :);
phys.LA50 = glb.LA50(iname_glb, :);
phys.Leq = glb.Leq(iname_glb, :);
phys.L10 = glb.L10(iname_glb, :);
phys.L50 = glb.L50(iname_glb, :);
phys.L90 = glb.L90(iname_glb, :);
phys.L50_1k = glb.L50_1k(iname_glb, :);
phys.TFSD_500 = glb.TFSD_500(iname_glb, :);
phys.TFSD_4k = glb.TFSD_4k(iname_glb, :);

[~, iname_src] = sort(src.name);
assert(isequal(iname_src, iname_glb), 'Different order');
phys.Leq_s = src.Leq_s(iname_src, :);
phys.Leq_o = src.Leq_o(iname_src, :);
phys.emergence = src.emergence(iname_src, :);
phys.t_pres_tf = src.t_pres_tf(iname_src, :);

load sim_glb;
load sim_src;

[~, iname_glb] = sort(glb.name);
phys.name(20:94, :) = glb.name(iname_glb, :);
phys.type(20:94, :) = glb.type(iname_glb, :);
phys.playbackLeq(20:94, :) = glb.playbackLeq(iname_glb, :);
phys.LAeq(20:94, :) = glb.LAeq(iname_glb, :);
phys.LA50(20:94, :) = glb.LA50(iname_glb, :);
phys.Leq(20:94, :) = glb.Leq(iname_glb, :);
phys.L10(20:94, :) = glb.L10(iname_glb, :);
phys.L50(20:94, :) = glb.L50(iname_glb, :);
phys.L90(20:94, :) = glb.L90(iname_glb, :);
phys.L50_1k(20:94, :) = glb.L50_1k(iname_glb, :);
phys.TFSD_500(20:94, :) = glb.TFSD_500(iname_glb, :);
phys.TFSD_4k(20:94, :) = glb.TFSD_4k(iname_glb, :);

[~, iname_src] = sort(src.name);
assert(isequal(iname_src, iname_glb), 'Different order');
phys.Leq_s(20:94, :) = src.Leq_s(iname_src, :);
phys.Leq_o(20:94, :) = src.Leq_o(iname_src, :);
phys.emergence(20:94, :) = src.emergence(iname_src, :);
phys.t_pres_tf(20:94, :) = src.t_pres_tf(iname_src, :);

save('phys_ordered.mat', 'phys');
