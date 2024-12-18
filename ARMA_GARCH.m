clc; clear; close all;

%% ------------------------- Collect Data ---------------------------------

% Read the CSV file
X = readtable('cex4WindDataInterpolated.csv', 'Delimiter', ',', 'ReadVariableNames', true);
X = rmmissing(X); % Remove rows containing NaN's
X.t = hours(X.t - X.t(1));

% Extract relevant columns
windPower = X.p;   % Measured wind power

% Number of observations
n = length(windPower);

% Define the split index
splitIndex = n - 2000;

% Split into training and testing sets
trainWindPower = windPower(1:splitIndex);
testWindPower = windPower(splitIndex + 1:end);
testTime = X.t(splitIndex + 1:end);

%% ------------------------- Specify ARMA-GARCH Model ---------------------

% Define ARMA(p,q) order
p = 4; % AR order 4
q = 4; % MA order 4

% Specify ARMA-GARCH model structure
Mdl = arima('ARLags', 1:p, 'MALags', 1:q, 'Variance', garch(1, 1));

% Fit the ARMA-GARCH model to the training data
EstMdl = estimate(Mdl, trainWindPower);

% Display estimated parameters
disp('Estimated ARMA-GARCH Parameters:');
disp(EstMdl);


%% ------------------------- Iterative Multi-Step Forecasting -------------

% Initialize storage for multi-step predictions
y_pred = zeros(length(testWindPower), 3); % Columns for 1-step, 2-step, and 3-step forecasts

% Combine training and test data for recursive predictions
allData = [trainWindPower; testWindPower];

% Perform iterative forecasting
for t = splitIndex + 1:n
    % 1-step prediction
    y_pred(t - splitIndex, 1) = forecast(EstMdl, 1, 'Y0', allData(1:t-1));

    % 2-step prediction
    if t - splitIndex > 1
        % Get 2-step forecast (returns a 2-element vector; extract the 2nd step)
        multiStepForecast = forecast(EstMdl, 2, 'Y0', allData(1:t-2));
        y_pred(t - splitIndex, 2) = multiStepForecast(2); % Extract the 2nd step
    end

    % 3-step prediction
    if t - splitIndex > 2
        % Get 3-step forecast (returns a 3-element vector; extract the 3rd step)
        multiStepForecast = forecast(EstMdl, 3, 'Y0', allData(1:t-3));
        y_pred(t - splitIndex, 3) = multiStepForecast(3); % Extract the 3rd step
    end

    % Ensure no negative predictions
    y_pred(t - splitIndex, :) = max(y_pred(t - splitIndex, :), 0);
end


%% ------------------------- Metrics Calculation --------------------------

% Define data used for evaluation of predictions
y_pred_eval = y_pred(end-1000+1:end, :);
testWindPower_eval = testWindPower(end-1000+1:end);
time_eval = testTime(end-1000+1:end);

% Initialize arrays to store metrics
RMSE = zeros(1, 3);
AIC = zeros(1, 3);
BIC = zeros(1, 3);

% Calculate residuals and metrics for 1-step, 2-step, and 3-step forecasts
for i = 1:3


    % Calculate residuals
    residuals = testWindPower_eval - y_pred_eval(:, i);
    RSS = sum(residuals.^2);
    N = length(testWindPower_eval);
    k = numel(EstMdl.AR) + numel(EstMdl.MA) + 1; % ARMA + GARCH params
    
    RMSE(i) = sqrt(RSS / N);
    AIC(i) = N * log(RSS / N) + 2 * k;
    BIC(i) = N * log(RSS / N) + k * log(N);

    % Display metrics
    fprintf('Step-%d Predictions:\n', i);
    fprintf('  RMSE: %.4f\n', RMSE(i));
    fprintf('  AIC: %.4f\n', AIC(i));
    fprintf('  BIC: %.4f\n\n', BIC(i));
end

%% ------------------------- Plot Predictions ----------------------------

function f = plotPredictions(y_true, y_pred, time, titleName)
    % Extract the last 101 points (assuming at least 101 data points)
    y_true_window = y_true(end-100:end);
    y_pred_window = y_pred(end-100:end, :);
    time_window = time(end-100:end);

    f = figure('Units', 'pixels', 'Position', [600, 300, 800, 600]);
    tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    for i = 1:3

        ax = nexttile;
        hold on;
        plot(time_window, y_true_window, 'DisplayName', 'Actual', 'LineWidth', 2, 'Color', 'k');
        plot(time_window, y_pred_window(:, i), 'DisplayName', 'Predicted', 'LineWidth', 1.5, 'Color', 'r', 'LineStyle', '--');
        
        if i == 1
            legend('Location','northwest', 'Interpreter','latex');
        end
        title(titleName{i}, 'Interpreter', 'latex');
        xlabel('Time', 'Interpreter','latex');
        ylabel('Wind Power','Interpreter','latex');
        xlim([time_window(1) time_window(end)]);
        grid on;
        box on;
        hold off;
        set(ax, 'FontSize', 13, 'TickLabelInterpreter', 'latex');
    end
end



% Plot predictions dynamically
titles = {'ARMA(4,4)-GARCH(1,1) - 1-step Predictions', ...
          'ARMA(4,4)-GARCH(1,1) - 2-step Predictions', ...
          'ARMA(4,4)-GARCH(1,1) - 3-step Predictions'};
f = plotPredictions(testWindPower_eval, y_pred_eval, time_eval, titles);

% Save plot as a PDF
exportgraphics(f,'ARMA_GARCH_pred.pdf', 'ContentType', 'vector');
