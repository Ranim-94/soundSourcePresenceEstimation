clear all, close all, clc;

dataset = {'train', 'test'};
dataset_path = {'../python/data/train/sound/', '../python/data/test/sound/'};

sources = {'T', 'V', 'B'};
source_names{1} = {'traffic', 'cityCar', 'roadCar'};
source_names{2} = {'crowd', 'voice'};
source_names{3} = {'bird'};

sr = 32e3;
l_frame = 32768;
[spl, freq] = iso226(0);
diff_Leq = 103.454;

alpha_opt = -14;
beta_opt = -7;
for i_dataset = 1:length(dataset)
    disp(['----- DATASET ' dataset{i_dataset} ' -----']);
    a_path = dir(dataset_path{i_dataset});
    a_path = {a_path(~[a_path.isdir]).name};
    a_path = a_path(cellfun('isempty', strfind(a_path, 'channel')));
    
    for i_a = 1:length(a_path)
        a_path_temp = a_path{i_a};
        disp(a_path_temp);
        
        %% Info
        src.name{i_a, 1} = a_path_temp(1:end-4);
        src.type{i_a, 1} = a_path_temp(1:end-6);
        
        %% Leq global (for hearing threshold)
        [x, sr_x] = audioread([dataset_path{i_dataset} a_path_temp]);
        x = resample(x, sr, sr_x);
        
        %% Channels
        c_path = dir([dataset_path{i_dataset} src.name{i_a} '*']);
        c_path = c_path(2:end); % Remove mix
        c_path = {c_path.name};
        
        for i_s = 1:length(source_names) % Source (T, V, B)
            disp([' -> ' sources{i_s}])
            i_path_s = zeros(size(c_path));
            for i_ss = 1:length(source_names{i_s}) % Subsource (traffic, cityCar, ...)
                % Indices of channels associated to current sources
                i_path_s = i_path_s | not(cellfun('isempty', strfind(c_path, source_names{i_s}{i_ss})));
            end
            c_path_s = c_path(i_path_s); % List of channels associated to current sources
            c_path_o = c_path(~i_path_s); % List of channels for other sources
            if ~isempty(c_path_s)
                %% Current source
                x = 0;
                for i_c = 1:length(c_path_s)
                    disp(['    S: ' c_path_s{i_c}]);
                    [x_c, sr_c] = audioread([dataset_path{i_dataset} c_path_s{i_c}]);
                    x_c = resample(x_c, sr, sr_c);
                    x = x + x_c;
                end
                
                % Sound level
                n_x = floor(length(x)/l_frame);
                L_tob_n = zeros(29, n_x);
                for indf = 1:n_x
                    if ~isempty(find(x((indf-1)*l_frame+1:indf*l_frame), 1))
                        xf = itaAudio(x((indf-1)*l_frame+1:indf*l_frame), sr, 'time');
                        X = ita_spk2frequencybands(xf,'mode','filter');
                        L_tob_n(:, indf) = X.freq(:, 1);
                    else
                        L_tob_n(:, indf) = 0;
                    end
                end
                % Leq
                L_tob_s = 20*log10(L_tob_n)+diff_Leq;
                L_s = 20*log10(sqrt(sum(L_tob_n.^2)))+diff_Leq;
                
                
                %% Other sources
                x_o = zeros(size(x));
                for i_c = 1:length(c_path_o)
                    disp(['    O: ' c_path_o{i_c}]);
                    [x_c, sr_c] = audioread([dataset_path{i_dataset} c_path_o{i_c}]);
                    x_c = resample(x_c, sr, sr_c);
                    x_o = x_o + x_c;
                end
                
                % Sound level
                n_x = floor(length(x_o)/l_frame);
                L_tob_n = zeros(29, n_x);
                for indf = 1:n_x
                    if ~isempty(find(x_o((indf-1)*l_frame+1:indf*l_frame), 1))
                        xf = itaAudio(x_o((indf-1)*l_frame+1:indf*l_frame), sr, 'time');
                        X = ita_spk2frequencybands(xf,'mode','filter');
                        L_tob_n(:, indf) = X.freq(:, 1);
                    else
                        L_tob_n(:, indf) = 0;
                    end
                end
                % Leq
                L_tob_o = 20*log10(L_tob_n)+diff_Leq;
                L_o = 20*log10(sqrt(sum(L_tob_n.^2)))+diff_Leq;
            else
                L_tob_s = -Inf;
                L_s = -Inf;
                L_tob_o = 0;
                L_o = 0;
            end
            [~, pres_tf] = timePresence(L_tob_s, L_tob_o, alpha_opt, beta_opt, spl', 0);
            csvwrite(['../python/data/' dataset{i_dataset} '/pres_profile/' src.name{i_a, 1} '_pp_' sources{i_s} '.csv'], pres_tf)
        end
    end
end

