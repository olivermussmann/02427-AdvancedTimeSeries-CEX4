clc; clear; close all;

%% ------------------------- Collect data ---------------------------------

% Read the CSV file
X = readtable('cex4WindDataInterpolated.csv', 'Delimiter', ',', 'ReadVariableNames', true);
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

% Function to calculate RMSE
calculateRMSE = @(y_true, y_pred) sqrt(mean((y_true - y_pred).^2));

% Function to calculate AIC
calculateAIC = @(RSS, numParams, numObs) 2 * numParams + numObs * log(RSS / numObs);

% Function to calculate BIC
calculateBIC = @(RSS, numParams, numObs) numObs * log(RSS / numObs) + numParams * log(numObs);

% Function to plot predictions vs actual
function f = plotPredictions(y_true, y_pred, time, titleName)
    
    data = {y_true(end-100:end), y_pred(end-100:end, 1), y_true(end-100:end), y_pred(end-100:end, 2), ...
        y_true(end-100:end), y_pred(end-100:end, 3)};
    
    

    f = figure('Units', 'pixels', 'Position', [600, 300, 800, 600]);
    tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    for i = 1:3
        ax = nexttile;
        hold on;
        plot(time, data{2*i-1}, 'b', 'DisplayName', 'Actual', 'LineWidth', 2, 'Color', 'k');
        plot(time, data{2*i}, 'r', 'DisplayName', 'Predicted', 'LineWidth', 1.5, 'Color', 'r', 'LineStyle', '--');
        if i == 1
            legend('Location','northwest', 'Interpreter','latex');
        end
        title(titleName{i}, 'Interpreter', 'latex');
        xlabel('Time Index', 'Interpreter','latex');
        ylabel('Wind Power','Interpreter','latex');
        xlim([time(1) time(end)])
        grid on;
        box on;
        hold off;
        set(ax, 'FontSize', 13, 'TickLabelInterpreter', 'latex')
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


% Define steps and parameters
steps = 1:3;
nSteps = length(steps);
nData = length(testWindPower);
params = 2; % Number of parameters: AR(1) + variance

% Compute residuals for all steps at once
residuals_AR1 = testWindPower - y_pred_AR1;

% Compute RSS, RMSE, AIC, and BIC in a vectorized manner
RSS_AR1 = sum(residuals_AR1.^2, 1);
RMSE_AR1 = sqrt(mean(residuals_AR1.^2, 1));
AIC_AR1 = nData * log(RSS_AR1 / nData) + 2 * params;
BIC_AR1 = nData * log(RSS_AR1 / nData) + params * log(nData);

% Display metrics using a loop for compactness
metrics = ["RMSE", "AIC", "BIC"];
for step = steps
    disp(['Metrics of ', num2str(step), '-step predictions']);
    disp(['AR(1) RMSE: ', num2str(RMSE_AR1(step))]);
    disp(['AR(1) AIC: ', num2str(AIC_AR1(step))]);
    disp(['AR(1) BIC: ', num2str(BIC_AR1(step))]);
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


% Define steps and parameters
steps = 1:3;
nSteps = length(steps);
nData = length(testWindPower);
params = 5; % Number of parameters: AR(4) + variance

% Compute residuals for all steps at once
residuals_AR4 = testWindPower - y_pred_AR4;

% Compute RSS, RMSE, AIC, and BIC in a vectorized manner
RSS_AR4 = sum(residuals_AR4.^2, 1);
RMSE_AR4 = sqrt(mean(residuals_AR4.^2, 1));
AIC_AR4 = nData * log(RSS_AR4 / nData) + 2 * params;
BIC_AR4 = nData * log(RSS_AR4 / nData) + params * log(nData);

% Display metrics using a loop for compactness
metrics = ["RMSE", "AIC", "BIC"];
for step = steps
    disp(['Metrics of ', num2str(step), '-step predictions']);
    disp(['AR(4) RMSE: ', num2str(RMSE_AR4(step))]);
    disp(['AR(4) AIC: ', num2str(AIC_AR4(step))]);
    disp(['AR(4) BIC: ', num2str(BIC_AR4(step))]);
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

% Define steps and parameters
steps = 1:3;
nSteps = length(steps);
nData = length(testWindPower);
params = 8; % Number of parameters: 4 AR + 3 MA + variance

% Compute residuals for all steps at once
residuals_ARMA = testWindPower - y_pred_ARMA;

% Compute RSS, RMSE, AIC, and BIC in a vectorized manner
RSS_ARMA = sum(residuals_ARMA.^2, 1);
RMSE_ARMA = sqrt(mean(residuals_ARMA.^2, 1));
AIC_ARMA = nData * log(RSS_ARMA / nData) + 2 * params;
BIC_ARMA = nData * log(RSS_ARMA / nData) + params * log(nData);

% Display metrics using a loop for compactness
metrics = ["RMSE", "AIC", "BIC"];
for step = steps
    disp(['Metrics of ', num2str(step), '-step predictions']);
    disp(['ARMA(4,3) RMSE: ', num2str(RMSE_ARMA(step))]);
    disp(['ARMA(4,3) AIC: ', num2str(AIC_ARMA(step))]);
    disp(['ARMA(4,3) BIC: ', num2str(BIC_ARMA(step))]);
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