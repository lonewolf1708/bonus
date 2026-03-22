%% =========================================================================
%  Space Propulsion 2025-26 - Flipped Class Assignment
%  Analysis of BATES motor burning rate data and motor ballistic model
%
%  Based on the Bayern-Chemie (BC) method for data regression
%  Reference: READ-ME-FIRST-2025_26-V01.pdf
%
%  Workflow:
%   1. Load 27 pressure traces (9 batches x 3 nozzle configurations)
%   2. Apply BC method to each trace -> (p_eff, t_burn)
%   3. Compute burning rate: rb = web / t_burn
%   4. Fit Vieille's law: rb = a * p^n  (via Uncertainty.m)
%   5. Compute experimental c* for each trace
%   6. Run quasi-steady motor model using Eq. 2 from README
% =========================================================================

clear; clc; close all;

%% =========================================================================
%  SECTION 1: MOTOR GEOMETRY AND PROPELLANT PROPERTIES
%  (From Figure 1 and Table 2 of READ-ME-FIRST-2025_26-V01.pdf)
% =========================================================================

% --- BATES grain geometry (BARIA motor, Figure 1) ---
D_o   = 116e-3;     % [m]  Grain outer diameter
D_i0  = 66e-3;      % [m]  Initial grain inner bore diameter
L_g   = 184e-3;     % [m]  Length of each grain
N_g   = 2;          % [-]  Number of grains
web   = (D_o - D_i0) / 2;   % [m]  Web thickness = 25 mm

fprintf('Motor geometry:\n');
fprintf('  D_outer = %.0f mm,  D_inner = %.0f mm,  web = %.0f mm\n', ...
        D_o*1e3, D_i0*1e3, web*1e3);
fprintf('  Grain length = %.0f mm,  N grains = %d\n', L_g*1e3, N_g);

% --- Propellant composition and density (Table 2) ---
% AP 68%, Al 18%, HTPB 14%
rho_AP   = 1950;    % [kg/m³]  Ammonium Perchlorate
rho_Al   = 2700;    % [kg/m³]  Aluminum
rho_HTPB =  920;    % [kg/m³]  HTPB binder
rho_p = 0.68*rho_AP + 0.18*rho_Al + 0.14*rho_HTPB;  % [kg/m³]
fprintf('  Propellant density = %.1f kg/m³\n', rho_p);

% --- Total propellant mass (assuming full web burnout) ---
M_prop = N_g * rho_p * pi/4 * (D_o^2 - D_i0^2) * L_g;  % [kg]
fprintf('  Total propellant mass = %.3f kg\n\n', M_prop);

% --- Nozzle throat areas (Table 1) ---
Dt_list    = [28.80e-3, 25.26e-3, 21.81e-3];  % [m]  Low / Mid / High
At_list    = pi * (Dt_list/2).^2;              % [m²]
label_list = {'Low pressure', 'Mid pressure', 'High pressure'};

fprintf('Nozzle configurations:\n');
for k = 1:3
    fprintf('  %s: D_t = %.2f mm,  A_t = %.4e m²\n', ...
            label_list{k}, Dt_list(k)*1e3, At_list(k));
end
fprintf('\n');

% --- Sampling frequency ---
fs = 1000;   % [Hz]
dt = 1/fs;   % [s]

% --- BC method parameters ---
pre_ign_window = round(0.020 * fs);  % 20 ms window to detect pre-ignition dip

% --- Motor model simulation parameters ---
dt_sim = 1e-4;   % [s]  simulation time step  (0.1 ms)
t_max  = 6.0;    % [s]  maximum simulation duration (well beyond expected burn time)

%% =========================================================================
%  SECTION 2: LOAD PRESSURE DATA
% =========================================================================

load('tracesbar1.mat');

% Batch names (9 batches)
% Batch variable names as loaded from tracesbar1.mat
% (names are assigned by the data acquisition system: pbar<batch_number>)
batch_names = {'pbar2438','pbar2439','pbar2440','pbar2441','pbar2442', ...
               'pbar2443','pbar2444','pbar2445','pbar2446'};
N_batches = length(batch_names);
N_cols    = 3;   % Low / Mid / High pressure per batch
N_traces  = N_batches * N_cols;   % 27 total firings

%% =========================================================================
%  SECTION 3: BC (Bayern-Chemie) METHOD - Apply to all 27 traces
%
%  Procedure (BC method, Paper-4-BC-V01.pdf):
%    A, G  = instants when pressure crosses 1% of p_max
%    I1    = integral of p from t_A to t_G
%    p_ref = I1 / (t_G - t_A)
%    B, E  = instants when pressure crosses p_ref
%    t_burn = t_E - t_B
%    p_eff  = integral(p, t_B, t_E) / t_burn
%    rb     = web / t_burn
% =========================================================================

% Pre-allocate result arrays
p_eff_all  = zeros(N_traces, 1);   % [bar]   Effective pressure
rb_all     = zeros(N_traces, 1);   % [mm/s]  Burning rate
tburn_all  = zeros(N_traces, 1);   % [s]     Burning time
At_all     = zeros(N_traces, 1);   % [m²]    Throat area for this firing
Ipc_all    = zeros(N_traces, 1);   % [N·s]   Pressure-impulse integral (for c*)

trace_id   = 0;

for b = 1:N_batches
    % Load the pressure data matrix for this batch [num_samples x 3]
    P_batch    = eval(batch_names{b});
    num_samples = size(P_batch, 1);
    t       = (0:num_samples-1)' * dt;    % time vector [s]

    % --- Determine which column uses which nozzle ---
    % Rank the columns by their mean pressure in the middle 50% of the trace
    % (lowest mean -> largest throat; highest mean -> smallest throat)
    mid_range = floor(num_samples/4) : floor(3*num_samples/4);
    col_means = mean(P_batch(mid_range, :));
    [~, rank]  = sort(col_means);   % rank(1)=col with lowest mean, etc.
    % Assign throat areas: rank 1 -> At_low, rank 2 -> At_mid, rank 3 -> At_high
    At_col = zeros(1, N_cols);
    for k = 1:N_cols
        At_col(rank(k)) = At_list(k);
    end

    for c = 1:N_cols
        trace_id = trace_id + 1;
        p = P_batch(:, c);   % pressure trace [bar]

        % ---- BC Method ----
        % Estimate atmospheric baseline from minimum in the pre-ignition window
        % (corresponds to the brief pressure drop just before ignition)
        win_end  = min(pre_ign_window, num_samples);
        p_atm    = min(p(1:win_end));
        p_max = max(p);
        p_max_gauge = p_max - p_atm;

        % 1% threshold above atmospheric
        thr = p_atm + 0.01 * p_max_gauge;

        % Find pre-ignition local minimum (the dip just before ignition)
        [~, idx_dip] = min(p(1:win_end));

        % idx_A: first point >= threshold AFTER the pre-ignition dip
        above_thr_after_dip = find(p(idx_dip:end) >= thr, 1, 'first');
        if isempty(above_thr_after_dip)
            warning('BC: cannot find point A for %s col %d', batch_names{b}, c);
            continue;
        end
        idx_A = above_thr_after_dip + idx_dip - 1;

        % idx_G: last point >= threshold in the full trace
        above_thr_all = find(p >= thr);
        idx_G = above_thr_all(end);

        t_A = t(idx_A);
        t_G = t(idx_G);

        % Integral I1 and reference pressure p_ref
        p_AG = p(idx_A:idx_G);
        t_AG = t(idx_A:idx_G);
        I1    = trapz(t_AG, p_AG);              % [bar·s]
        p_ref = I1 / (t_G - t_A);              % [bar]

        % B and E: first and last index where p >= p_ref in [A,G]
        above_pref_AG = find(p_AG >= p_ref);
        if isempty(above_pref_AG)
            warning('BC: cannot find points B/E for %s col %d', batch_names{b}, c);
            continue;
        end
        idx_B = above_pref_AG(1)   + idx_A - 1;
        idx_E = above_pref_AG(end) + idx_A - 1;

        t_B    = t(idx_B);
        t_E    = t(idx_E);
        t_burn = t_E - t_B;           % [s]

        % Effective pressure
        p_BE  = p(idx_B:idx_E);
        t_BE  = t(idx_B:idx_E);
        I_BE  = trapz(t_BE, p_BE);    % [bar·s]
        p_eff = I_BE / t_burn;        % [bar]

        % Burning rate
        rb = (web * 1e3) / t_burn;    % [mm/s]  (web in m -> *1e3 for mm)

        % Pressure impulse for c* (convert pressure to Pa: 1 bar = 1e5 Pa)
        Ipc = I_BE * 1e5 * At_col(c);  % [Pa·s·m²] = [N·s]

        % Store results
        p_eff_all(trace_id) = p_eff;
        rb_all(trace_id)    = rb;
        tburn_all(trace_id) = t_burn;
        At_all(trace_id)    = At_col(c);
        Ipc_all(trace_id)   = Ipc;
    end
end

fprintf('BC Method Results (all 27 traces):\n');
fprintf('  p_eff range: %.1f - %.1f bar\n', min(p_eff_all), max(p_eff_all));
fprintf('  t_burn range: %.3f - %.3f s\n',  min(tburn_all), max(tburn_all));
fprintf('  rb range:    %.2f - %.2f mm/s\n', min(rb_all),   max(rb_all));
fprintf('\n');

%% =========================================================================
%  SECTION 4: VIEILLE'S LAW CHARACTERIZATION
%  rb = a * p_eff^n  (fit via Uncertainty.m)
% =========================================================================

fprintf('=== Vieille''s Law Characterization ===\n');
[a, Inc_a, n_exp, Inc_n, R2] = Uncertainty(p_eff_all, rb_all);

fprintf('  a      = %.4f ± %.4f  mm/s/bar^n\n', a, Inc_a);
fprintf('  n      = %.4f ± %.4f\n',              n_exp, Inc_n);
fprintf('  R²     = %.6f\n\n',                   R2);

%% =========================================================================
%  SECTION 5: EXPERIMENTAL c* (CHARACTERISTIC VELOCITY)
%
%  C*_exp = integral(p_c * A_t, t_B, t_E) / M_prop   [Eq. 1, README]
% =========================================================================

cstar_all = Ipc_all ./ M_prop;   % [m/s]

cstar_mean = mean(cstar_all);
cstar_std  = std(cstar_all);

fprintf('=== Experimental c* ===\n');
fprintf('  c*_mean = %.1f m/s\n',  cstar_mean);
fprintf('  c*_std  = %.1f m/s  (±%.1f%%)\n', cstar_std, 100*cstar_std/cstar_mean);
fprintf('\n');

%% =========================================================================
%  SECTION 6: PLOTS
% =========================================================================

% --- Plot 1: Burning rate vs effective pressure (log-log with Vieille fit) ---
figure(1); clf;
loglog(p_eff_all, rb_all, 'bo', 'MarkerSize', 7, 'MarkerFaceColor', [0.5 0.7 1], ...
       'DisplayName', 'Experimental data (27 firings)');
hold on;
p_fit  = linspace(min(p_eff_all)*0.9, max(p_eff_all)*1.1, 200);
rb_fit = a * p_fit.^n_exp;
loglog(p_fit, rb_fit, 'r-', 'LineWidth', 2.5, ...
       'DisplayName', sprintf('Vieille fit: r_b = %.3f \\cdot p^{%.3f}  (R^2=%.4f)', ...
       a, n_exp, R2));
xlabel('Effective Chamber Pressure  [bar]', 'FontSize', 12);
ylabel('Burning Rate  [mm/s]',              'FontSize', 12);
title('Vieille''s Law: r_b = a \cdot p^n',  'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 11);
grid on; grid minor;
text(min(p_eff_all)*1.05, max(rb_all)*0.95, ...
     sprintf('a = %.3f ± %.3f mm/s/bar^n\nn = %.3f ± %.3f\nR^2 = %.4f', ...
     a, Inc_a, n_exp, Inc_n, R2), ...
     'FontSize', 10, 'VerticalAlignment', 'top');

% --- Plot 2: c* per trace ---
figure(2); clf;
bar(cstar_all, 'FaceColor', [0.6 0.8 0.6], 'EdgeColor', [0.2 0.5 0.2]);
hold on;
yline(cstar_mean, 'r-', 'LineWidth', 2, 'DisplayName', ...
      sprintf('Mean c* = %.0f m/s', cstar_mean));
yline(cstar_mean + cstar_std, 'r--', 'LineWidth', 1, ...
      'DisplayName', sprintf('\\pm1\\sigma = ±%.0f m/s', cstar_std));
yline(cstar_mean - cstar_std, 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
xlabel('Trace index (1-27)',       'FontSize', 12);
ylabel('Experimental c*  [m/s]', 'FontSize', 12);
title('Experimental Characteristic Velocity c*', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'south', 'FontSize', 11);
ylim([cstar_mean*0.8, cstar_mean*1.2]);
grid on;

%% =========================================================================
%  SECTION 7: QUASI-STEADY MOTOR MODEL
%
%  Quasi-steady SRM pressure:
%    p(t) = (a * rho_p * c* * Ab(r)/At)^(1/(1-n))   [Eq. 2, README]
%
%  Time integration:
%    dr/dt = r_b(t) = a_SI * p(t)^n_exp
%    r(0)  = D_i0 / 2
%    Stop when r >= D_o/2  (web fully consumed)
% =========================================================================

fprintf('=== Motor Model Simulation ===\n');

% --- Select nozzle configuration ---
%  Options: 'low', 'mid', 'high'  (refers to pressure level)
throat_choice = 'mid';

switch lower(throat_choice)
    case 'low'
        At_model = At_list(1);
        Dt_label = sprintf('Low pressure (D_t = %.2f mm)', Dt_list(1)*1e3);
    case 'mid'
        At_model = At_list(2);
        Dt_label = sprintf('Mid pressure (D_t = %.2f mm)', Dt_list(2)*1e3);
    case 'high'
        At_model = At_list(3);
        Dt_label = sprintf('High pressure (D_t = %.2f mm)', Dt_list(3)*1e3);
    otherwise
        error('throat_choice must be ''low'', ''mid'', or ''high''');
end

% --- Convert Vieille's law to SI ---
% a [mm/s/bar^n] -> a_SI [m/s/Pa^n]:
%   rb[m/s] = a_SI * p[Pa]^n
%   a_SI = a[mm/s/bar^n] * 1e-3 / (1e5)^n
a_SI = a * 1e-3 / (1e5)^n_exp;    % [m/s/Pa^n]

% Use mean experimental c*
cstar_model = cstar_mean;   % [m/s]

% --- Time integration (explicit Euler) ---
t_sim    = 0 : dt_sim : t_max;
N_sim    = length(t_sim);
p_model  = zeros(1, N_sim);    % [Pa]
r_bore   = zeros(1, N_sim);    % [m]  bore radius over time
r_bore(1) = D_i0 / 2;         % initial bore radius [m]

R_outer  = D_o / 2;            % outer grain radius [m]
t_burnout = t_max;

for k = 1:N_sim-1
    r = r_bore(k);

    % Burning surface area (BATES: cylindrical bore + end annuli, N_g grains)
    Ab_cyl  = N_g * 2*pi * r * L_g;                    % [m²]
    Ab_ends = N_g * 2 * (pi/4) * (R_outer^2 - r^2);    % [m²]
    Ab_k    = Ab_cyl + Ab_ends;                          % [m²]

    % Quasi-steady pressure [Pa]  (Eq. 2)
    p_k = (a_SI * rho_p * cstar_model * Ab_k / At_model)^(1/(1-n_exp));
    p_model(k) = p_k;

    % Bore radius advance
    rb_k     = a_SI * p_k^n_exp;             % [m/s]
    r_bore(k+1) = r + rb_k * dt_sim;         % [m]

    % Check web burnout
    if r_bore(k+1) >= R_outer
        p_model(k+1:end) = 0;
        t_burnout = t_sim(k+1);
        break;
    end
end

% Convert model pressure to bar
p_model_bar = p_model * 1e-5;   % [bar]

% Compute burn time and mean pressure of the simulation
above_1pct = find(p_model_bar > 0.01 * max(p_model_bar));
if ~isempty(above_1pct)
    t_start_model = t_sim(above_1pct(1));
    t_end_model   = t_sim(above_1pct(end));
    t_burn_model  = t_end_model - t_start_model;
    p_mean_model  = mean(p_model_bar(above_1pct));
else
    t_burn_model = 0; p_mean_model = 0;
end

fprintf('  Nozzle: %s\n', Dt_label);
fprintf('  Burnout at t = %.3f s\n', t_burnout);
fprintf('  Burn time (model) = %.3f s\n', t_burn_model);
fprintf('  Mean chamber pressure (model) = %.2f bar\n\n', p_mean_model);

% --- Plot 3: Model pressure trace ---
figure(3); clf;
plot(t_sim, p_model_bar, 'b-', 'LineWidth', 2);
xlabel('Time  [s]',          'FontSize', 12);
ylabel('Chamber Pressure  [bar]', 'FontSize', 12);
title(['Quasi-steady Motor Model: ' Dt_label], 'FontSize', 13, 'FontWeight', 'bold');
grid on;
text(t_sim(end)*0.05, max(p_model_bar)*0.95, ...
     sprintf('a = %.3f mm/s/bar^n\nn = %.3f\nc* = %.0f m/s\nt_{burn} = %.2f s', ...
     a, n_exp, cstar_model, t_burn_model), ...
     'FontSize', 10, 'VerticalAlignment', 'top', 'BackgroundColor', 'w');

% --- Plot 4: Bore radius evolution ---
figure(4); clf;
t_bore_valid = t_sim(r_bore > 0);
r_bore_mm    = r_bore(r_bore > 0) * 1e3;   % convert to mm
% Trim to burn duration
idx_end_bore = find(t_bore_valid >= t_burnout, 1, 'first');
if ~isempty(idx_end_bore)
    t_bore_valid = t_bore_valid(1:idx_end_bore);
    r_bore_mm    = r_bore_mm(1:idx_end_bore);
end
plot(t_bore_valid, r_bore_mm, 'm-', 'LineWidth', 2);
hold on;
yline(R_outer*1e3, 'k--', 'LineWidth', 1.5, 'Label', 'Outer radius');
xlabel('Time  [s]', 'FontSize', 12);
ylabel('Bore Radius  [mm]', 'FontSize', 12);
title('Bore Radius vs Time (Motor Model)', 'FontSize', 13, 'FontWeight', 'bold');
grid on;

%% =========================================================================
%  SUMMARY
% =========================================================================

fprintf('========================================\n');
fprintf('  SUMMARY OF RESULTS\n');
fprintf('========================================\n');
fprintf('  Vieille''s law:  r_b = a * p^n\n');
fprintf('    a = %.4f ± %.4f mm/s/bar^n\n', a, Inc_a);
fprintf('    n = %.4f ± %.4f\n',             n_exp, Inc_n);
fprintf('    R² = %.6f\n',                   R2);
fprintf('\n');
fprintf('  Characteristic velocity c*:\n');
fprintf('    Mean  = %.1f m/s\n',  cstar_mean);
fprintf('    Std   = %.1f m/s  (±%.1f%%)\n', cstar_std, 100*cstar_std/cstar_mean);
fprintf('\n');
fprintf('  Motor model (%s):\n', Dt_label);
fprintf('    Predicted burn time = %.3f s\n', t_burn_model);
fprintf('    Mean chamber pressure = %.2f bar\n', p_mean_model);
fprintf('========================================\n');
