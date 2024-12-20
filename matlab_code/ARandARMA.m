clc; clear; close all;

%% ------------------------- Collect data ---------------------------------

% Read the CSV file
X = readtable('C:\Users\lucas\Documents\GitHub\02427-AdvancedTimeSeries-CEX4\comp_ex_4_scripts_2011\data\cex4WindDataInterpolated.csv', 'Delimiter', ',', 'ReadVariableNames', true);
X = rmmissing(X); % Remove rows containing NaN's
X.t = hours(X.t - X.t(1));


% Extract the wind power data
windPower = X.p; % Measured wind power

% Number of observations
n = length(windPower);

% Define the split index
splitIndex = n - 1000;

% Split into training and testing sets
trainWindPower = windPower(1:splitIndex);
testWindPower = windPower(splitIndex + 1:end);

%% ------------------------- Helper Functions ----------------------------

function [RMSE, AIC, BIC] = calculateMetrics(y_true, y_pred, k)
    RMSE = zeros(1, 3);
    AIC = zeros(1, 3);
    BIC = zeros(1, 3);

    for i = 1:3
        % Valid indices for comparison
        validIdx = ~isnan(y_pred(:, i));
        y_true_valid = y_true(validIdx); % Where i is the step-ahead (1, 2, or 3)
        y_pred_valid = y_pred(validIdx, i);
        
        % Residuals
        residuals = y_true_valid - y_pred_valid;

        % Residual Sum of Squares
        RSS = sum(residuals.^2);
        N = length(y_true_valid);

        % Compute Metrics
        RMSE(i) = sqrt(RSS / N);
        AIC(i) = N * log(RSS / N) + 2 * k;
        BIC(i) = N * log(RSS / N) + k * log(N);
    end
end

function f = plotPredictions(y_true, y_pred, time, titleName)
    f = figure('Units', 'pixels', 'Position', [600, 300, 800, 600]);
    tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    for i = 1:3
        ax = nexttile;
        hold on;

        plot_length = length(time);
        time_valid = time(i:end);
        y_true_valid = y_true(end - plot_length:end-i);
        y_pred_valid = y_pred(end - plot_length+i:end, i);

        plot(time_valid, y_true_valid, 'k', 'DisplayName', 'Actual', 'LineWidth', 2);
        plot(time_valid, y_pred_valid, 'r--', 'DisplayName', 'Predicted', 'LineWidth', 1.5);
        
        title(titleName{i}, 'Interpreter', 'latex');
        xlabel('Time', 'Interpreter', 'latex');
        ylabel('Wind Power', 'Interpreter', 'latex');
        legend('Location', 'northwest', 'Interpreter', 'latex');
        grid on;
        hold off;
    end
end


%% ------------------------- AR(1) Model ----------------------------------

% Define AR(1) model
model_AR1 = arima('ARLags', 1, 'Constant', 0);

% Estimate AR(1) model
model_AR1_Fit = estimate(model_AR1, trainWindPower);

% Predictions for test data
y_pred_AR1 = zeros(length(testWindPower), 3);
for t = 1:length(testWindPower)
    % Forecast one step ahead using previous training data
    currentData = [trainWindPower; testWindPower(1:t-1)];
    
    y_pred_AR1(t, 1) = forecast(model_AR1_Fit, 1, 'Y0', currentData);
    
    % Check if we can store the second step prediction
    if t < length(testWindPower)
        pred2 = forecast(model_AR1_Fit, 2, 'Y0', currentData);
        y_pred_AR1(t+1, 2) = pred2(2);
    end
    
    % Check if we can store the third step prediction
    if t < length(testWindPower) - 1
        pred3 = forecast(model_AR1_Fit, 3, 'Y0', currentData);
        y_pred_AR1(t+2, 3) = pred3(3);
    end
    
    % Ensure predictions are non-negative
    if any(y_pred_AR1(t, :) < 0)
        y_pred_AR1(t, y_pred_AR1(t, :) < 0) = 0;
    end
end

% Define data used for evaluation of predictions
y_pred_eval = y_pred_AR1(end-1000+1:end, :);
testWindPower_eval = testWindPower(end-1000+1:end);

% Calculate and print metrics
[RMSE_AR1, AIC_AR1, BIC_AR1] = calculateMetrics(testWindPower_eval, y_pred_eval, model_AR1_Fit.P + model_AR1_Fit.Q + model_AR1_Fit.D + 1);

fprintf('Prediction Metrics:\n');
for i = 1:3
    fprintf('Step-%d Predictions:\n', i);
    fprintf('  RMSE: %.4f\n', RMSE_AR1(i));
    fprintf('  AIC: %.4f\n', AIC_AR1(i));
    fprintf('  BIC: %.4f\n', BIC_AR1(i));
end

% Plot predictions dynamically
titles = {'AR(1) - 1-step Predictions', 'AR(1) - 2-step Predictions', 'AR(1) - 3-step Predictions'};

f = plotPredictions(testWindPower, y_pred_AR1, X.t(end-100:end), titles);
exportgraphics(f,'AR1_pred.pdf', 'ContentType', 'vector')

%% ------------------------- AR(4) Model ----------------------------------

% Define AR(4) model
model_AR4 = arima('ARLags', 1:4, 'Constant', 0);

% Estimate AR(4) model
model_AR4_Fit = estimate(model_AR4, trainWindPower);

% Predictions for test data
y_pred_AR4 = zeros(length(testWindPower), 3);
for t = 1:length(testWindPower)
    % Forecast one step ahead using previous training data
    currentData = [trainWindPower; testWindPower(1:t-1)];
    
    y_pred_AR4(t, 1) = forecast(model_AR4_Fit, 1, 'Y0', currentData);
    
    % Check if we can store the second step prediction
    if t < length(testWindPower)
        pred2 = forecast(model_AR4_Fit, 2, 'Y0', currentData);
        y_pred_AR4(t+1, 2) = pred2(2);
    end
    
    % Check if we can store the third step prediction
    if t < length(testWindPower) - 1
        pred3 = forecast(model_AR4_Fit, 3, 'Y0', currentData);
        y_pred_AR4(t+2, 3) = pred3(3);
    end
    
    % Ensure predictions are non-negative
    if any(y_pred_AR4(t, :) < 0)
        y_pred_AR4(t, y_pred_AR4(t, :) < 0) = 0;
    end
end
% Define data used for evaluation of predictions
y_pred_eval = y_pred_AR4(end-1000+1:end, :);
testWindPower_eval = testWindPower(end-1000+1:end);

% Calculate and print metrics
[RMSE_AR4, AIC_AR4, BIC_AR4] = calculateMetrics(testWindPower_eval, y_pred_eval, model_AR4_Fit.P + model_AR4_Fit.Q + model_AR4_Fit.D + 1);

fprintf('Prediction Metrics:\n');
for i = 1:3
    fprintf('Step-%d Predictions:\n', i);
    fprintf('  RMSE: %.4f\n', RMSE_AR4(i));
    fprintf('  AIC: %.4f\n', AIC_AR4(i));
    fprintf('  BIC: %.4f\n', BIC_AR4(i));
end

% Plot predictions dynamically
titles = {'AR(4) - 1-step Predictions', 'AR(4) - 2-step Predictions', 'AR(4) - 3-step Predictions'};
f = plotPredictions(testWindPower, y_pred_AR4, X.t(end-100:end), titles);
exportgraphics(f,'AR4_pred.pdf', 'ContentType', 'vector')

%% ------------------------- ARMA(4,3) Model ------------------------------

% Define ARMA(4,3) model
model_ARMA = arima('ARLags', 1:4, 'MALags', 1:3, 'Constant', 0);

% Estimate ARMA(4,3) model
model_ARMA_Fit = estimate(model_ARMA, trainWindPower);

% Predictions for test data
y_pred_ARMA = zeros(length(testWindPower), 3);
for t = 1:length(testWindPower)
    % Forecast one step ahead using previous training data
    currentData = [trainWindPower; testWindPower(1:t-1)];
    y_pred_ARMA(t, 1) = forecast(model_ARMA_Fit, 1, 'Y0', currentData);
    
    % Check if we can store the second step prediction
    if t < length(testWindPower)
        pred2 = forecast(model_ARMA_Fit, 2, 'Y0', currentData);
        y_pred_ARMA(t+1, 2) = pred2(2);
    end
    
    % Check if we can store the third step prediction
    if t < length(testWindPower) - 1
        pred3 = forecast(model_ARMA_Fit, 3, 'Y0', currentData);
        y_pred_ARMA(t+2, 3) = pred3(3);
    end
    
    % Ensure predictions are non-negative
    if any(y_pred_ARMA(t, :) < 0)
        y_pred_ARMA(t, y_pred_ARMA(t, :) < 0) = 0;
    end
end

% Define data used for evaluation of predictions
y_pred_eval = y_pred_ARMA(end-1000+1:end, :);
testWindPower_eval = testWindPower(end-1000+1:end);

% Calculate and print metrics
[RMSE_ARMA, AIC_ARMA, BIC_ARMA] = calculateMetrics(testWindPower_eval, y_pred_eval, model_ARMA_Fit.P + model_ARMA_Fit.Q + model_ARMA_Fit.D + 1);

fprintf('Prediction Metrics:\n');
for i = 1:3
    fprintf('Step-%d Predictions:\n', i);
    fprintf('  RMSE: %.4f\n', RMSE_ARMA(i));
    fprintf('  AIC: %.4f\n', AIC_ARMA(i));
    fprintf('  BIC: %.4f\n', BIC_ARMA(i));
end

% Plot predictions dynamically
titles = {'ARMA(4,3) - One-step Predictions', 'ARMA(4,3) - Two-step Predictions', 'ARMA(4,3) - Three-step Predictions'};
f = plotPredictions(testWindPower, y_pred_ARMA, X.t(end-100:end), titles);
exportgraphics(f,'ARMA43_pred.pdf', 'ContentType', 'vector')

%% ------------------------- Results Summary ------------------------------

% Summarize results
disp('One-step prediction comparison')
results = table({'AR(1)'; 'AR(4)'; 'ARMA(4,3)'}, ...
                [RMSE_AR1(1); RMSE_AR4(1); RMSE_ARMA(1)], ...
                [AIC_AR1(1); AIC_AR4(1); AIC_ARMA(1)], ...
                [BIC_AR1(1); BIC_AR4(1); BIC_ARMA(1)], ...
                'VariableNames', {'Model', 'RMSE', 'AIC', 'BIC'});

disp(results);


%% Save script to Github

current_file = 'ARandARMA.m';
new_directory = '/home/olivermussmann/Documents/GitHub/02427-AdvancedTimeSeries-CEX4/matlab_code';
copyfile(current_file, new_directory);