function visualizeCorpus(dataset, colorData)
% Visualization function for the simulated part of the corpus
%   - Estimated perceptual times of presence are used as axis
%   - Clicking on a point plays the corresponding audio
%
% Input: 
%   - dataset:
%       'rep': Replicated scenes (Grafic)
%       'sim': Simulated scenes (simScene composition)
%   - colorData: data used for displayed points colors
%       'ambiance': original ambiance of the scene
%       'Leq': Playback Leq (dB)

if nargin<2; colorData='ambiance'; end;
if nargin<1; dataset='sim'; end;

switch dataset
    case 'rep'
        data_path = '../audio_scaled/rep/';
		load rep_exp;
	case 'sim'
		data_path = '../audio_scaled/sim/';
		load sim_exp;
	otherwise
		error('Unknown dataset.');
end

locs = exp_scenes.t_pres;
a_path = exp_scenes.name;

switch colorData
    case 'ambiance' % Original ambiance
        switch dataset
            case 'rep'
                colors = (~cellfun('isempty', strfind(exp_scenes.name, 'Park'))) + ...
                    2*(~cellfun('isempty', strfind(exp_scenes.name, 'QuietStreet'))) + ...
                    3*(~cellfun('isempty', strfind(exp_scenes.name, 'NoisyStreet'))) + ...
                    4*(~cellfun('isempty', strfind(exp_scenes.name, 'VeryNoisyStreet')));
            case 'sim'
                colors = (~cellfun('isempty', strfind(exp_scenes.name, 'park'))) + ...
                    2*(~cellfun('isempty', strfind(exp_scenes.name, 'quietStreet'))) + ...
                    3*(~cellfun('isempty', strfind(exp_scenes.name, 'noisyStreet'))) + ...
                    4*(~cellfun('isempty', strfind(exp_scenes.name, 'veryNoisyStreet'))) + ...
                    5*(~cellfun('isempty', strfind(exp_scenes.name, 'square')));
        end
        colormap(jet)
    case 'Leq' % Playback Leq
        colors = exp_scenes.Leq;
        colormap(jet)
    otherwise % Default Matlab color
        colors = [0 0.4470 0.7410];
end

color_sc = 'gmrkb';
color_sc = [0, 0.5, 0; ...
    0.5, 0, 0.5; ...
    1, 0, 0; ...
    0, 0, 0; ...
    0, 0, 1];
marker_sc = 'ovds*';

fontSize = 20;
set(0, 'defaultTextFontSize',fontSize)

figure(1), clf, 
a = gscatter(locs(:, 1), locs(:, 2), {colors(:, 1)}, color_sc, marker_sc, 10);
set(a(1), 'MarkerFaceColor', [0, 0.5, 0])
set(a(2), 'MarkerFaceColor', [0.5, 0, 0.5])
set(a(3), 'MarkerFaceColor', [1, 0, 0])
set(a(4), 'MarkerFaceColor', [0, 0, 0])
set(a(5), 'MarkerFaceColor', [0, 0, 1])
grid on, axis([0 1 0 1]), xlabel('Traffic', 'FontSize',fontSize), ylabel('Voices', 'FontSize',fontSize)
legend('Park', 'Quiet Street', 'Noisy Street', 'Very Noisy Street', 'Square', 'Location', 'Best')
set(gca, 'FontSize', fontSize)
export_fig tv_pres.eps -eps -transparent

figure(2), clf, 
a = gscatter(locs(:, 1), locs(:, 3), {colors(:, 1)}, color_sc, marker_sc, 10, 'filled');
set(a(1), 'MarkerFaceColor', [0, 0.5, 0])
set(a(2), 'MarkerFaceColor', [0.5, 0, 0.5])
set(a(3), 'MarkerFaceColor', [1, 0, 0])
set(a(4), 'MarkerFaceColor', [0, 0, 0])
set(a(5), 'MarkerFaceColor', [0, 0, 1])
grid on, axis([0 1 0 1]), xlabel('Traffic', 'FontSize',fontSize), ylabel('Birds', 'FontSize',fontSize)
set(gca, 'FontSize', fontSize)
export_fig tb_pres.eps -eps -transparent

figure(3), clf, 
a = gscatter(locs(:, 2), locs(:, 3), {colors(:, 1)}, color_sc, marker_sc, 10, 'filled');
set(a(1), 'MarkerFaceColor', [0, 0.5, 0])
set(a(2), 'MarkerFaceColor', [0.5, 0, 0.5])
set(a(3), 'MarkerFaceColor', [1, 0, 0])
set(a(4), 'MarkerFaceColor', [0, 0, 0])
set(a(5), 'MarkerFaceColor', [0, 0, 1])
grid on, axis([0 1 0 1]), xlabel('Voices', 'FontSize',fontSize), ylabel('Birds', 'FontSize',fontSize)
set(gca, 'FontSize', fontSize)
export_fig vb_pres.eps -eps -transparent


end