clc; clear; close all;

%% ------------------------- Collect data ---------------------------------

% Read the CSV file
X = readtable('C:\Users\lucas\Documents\GitHub\02427-AdvancedTimeSeries-CEX4\comp_ex_4_scripts_2011\data\cex4WindDataInterpolated.csv', 'Delimiter', ',', 'ReadVariableNames', true);
X = rmmissing(X); % Remove rows containing NaN's
X.t = hours(X.t - X.t(1));

% Extract relevant columns
windPower = X.p;   % Measured wind power
windSpeed = X.Ws1; % 1-hour forecasted wind speed (exogenous input)
windSpeed2 = X.Ws2; % 2-hour forecasted wind speed (exogenous input)
windSpeed3 = X.Ws3; % 3-hour forecasted wind speed (exogenous input)
windDir = X.Wd1;   % 1-hour forecasted wind direction (for regimes)
windDir2 = X.Wd2;   % 2-hour forecasted wind direction (for regimes)
windDir3 = X.Wd3;   % 3-hour forecasted wind direction (for regimes)

% Number of observations
n = length(windPower);

% Define the split index
splitIndex = n - 1000;

% Split into training and testing sets
trainWindPower = windPower(1:splitIndex);
trainWindSpeed = windSpeed(1:splitIndex);
trainWindDir = windDir(1:splitIndex);

testWindPower = windPower(splitIndex + 1:end);

testWindSpeed = windSpeed(splitIndex + 1:end);
testWindSpeed2 = windSpeed2(splitIndex + 1:end);
testWindSpeed3 = windSpeed3(splitIndex + 1:end);

testWindDir = windDir(splitIndex + 1:end);
testWindDir2 = windDir2(splitIndex + 1:end);
testWindDir3 = windDir3(splitIndex + 1:end);

testTime = X.t(splitIndex + 1:end);


%% ------------------------- Regime Definitions --------------------------

% Define the regimes
R1 = @(dir) (dir >= 300 | dir < 60);   % 300°-360° and 0°-60°
R2 = @(dir) (dir >= 60 & dir < 180);  % 60°-180°
R3 = @(dir) (dir >= 180 & dir < 300); % 180°-300°

% Identify regimes for training data
regimeTrain = zeros(size(trainWindDir));
regimeTrain(R1(trainWindDir)) = 1;
regimeTrain(R2(trainWindDir)) = 2;
regimeTrain(R3(trainWindDir)) = 3;

% Identify regimes for testing data
regimeTest = zeros(size(testWindDir));
regimeTest(R1(testWindDir)) = 1;
regimeTest(R2(testWindDir)) = 2;
regimeTest(R3(testWindDir)) = 3;

regimeTest2 = zeros(size(testWindDir2));
regimeTest2(R1(testWindDir2)) = 1;
regimeTest2(R2(testWindDir2)) = 2;
regimeTest2(R3(testWindDir2)) = 3;

regimeTest3 = zeros(size(testWindDir3));
regimeTest3(R1(testWindDir3)) = 1;
regimeTest3(R2(testWindDir3)) = 2;
regimeTest3(R3(testWindDir3)) = 3;

%% ------------------------- TARX Model Training -------------------------

% AR lag order
lagOrder = 5;
exoOrder = 3;

% Prepare data storage for models
models_TARX = cell(1, 3);

% Train TARX model for each regime
for regime = 1:3
    % Filter data for the current regime
    idx = (regimeTrain == regime);
    y_train = trainWindPower(idx);
    x_train = trainWindSpeed(idx);
    nTrain = length(y_train);
    
    % Construct lagged AR terms
    if nTrain > lagOrder
        X_AR = zeros(nTrain - lagOrder, lagOrder);  % AR lags
        X_X = x_train(lagOrder + 1:end);            % Exogenous variable
        
        for i = 1:lagOrder
            X_AR(:, i) = y_train(lagOrder + 1 - i:nTrain - i);
        end

        for i = 1:exoOrder
            X_X(:, i) = x_train(lagOrder + 1 - i:nTrain - i);
        end

        % Combine AR and exogenous terms
        X_combined = [X_AR, X_X];
        y_target = y_train(lagOrder + 1:end);
        
        % Fit linear model for the current regime
        models_TARX{regime} = fitlm(X_combined, y_target);

    end
end

%% ------------------------- TARX 1-Step Predictions ---------------------

% Predict for test data (one-step ahead)
y_pred_TARX = zeros(size(testWindPower, 3));
for t = lagOrder+1:length(testWindPower)
    % Determine the regime for the current observation
    currentRegime = regimeTest(t);

    % Use the model for the current regime
    mdl = models_TARX{currentRegime};
    
    % Ensure enough past values are available
    if t > lagOrder
        X_AR = flip(testWindPower(t - lagOrder:t - 1))'; % Use past actual values
    else
        % Use available values and pad with NaN for initial steps
        X_AR = [NaN(1, lagOrder - (t - 1)), flip(testWindPower(1:t - 1))'];
    end
    
    % Include the exogenous variable
    for i = 1:exoOrder
        X_X = flip(testWindSpeed(t-i+1:t))'; % Exogenous variable
    end
    X_combined = [X_AR, X_X];
    
    % One-step prediction
    y_pred_TARX(t, 1) = predict(mdl, X_combined);
    if y_pred_TARX(t, 1) < 0
        y_pred_TARX(t, 1) = 0;
    end

    % Two-step prediction
    if t < length(testWindPower) - 1
        % Determine the regime for the current observation
        currentRegime2 = regimeTest2(t+1);

        % Use the model for the current regime (2-hour predicted wind dir.)
        mdl = models_TARX{currentRegime2};
        X_AR_2 = [y_pred_TARX(t, 1), X_AR(1:end-1)];
        X_X_2 = [testWindSpeed2(t+1), X_X(1:end-1)];
        X_combined2 = [X_AR_2, X_X_2];
        y_pred_TARX(t+1, 2) = predict(mdl, X_combined2);
        if y_pred_TARX(t+1, 2) < 0
            y_pred_TARX(t+1, 2) = 0;
        end
    end

    % Three-step prediction
    if t < length(testWindPower) - 2
        % Determine the regime for the current observation
        currentRegime3 = regimeTest3(t+2);

        % Use the model for the current regime (3-hour predicted wind dir.)
        mdl = models_TARX{currentRegime3};

        X_AR_3 = [y_pred_TARX(t+1, 2), X_AR_2(1:end-1)];
        X_X_3 = [testWindSpeed3(t+2), X_X_2(1:end-1)];
        X_combined3 = [X_AR_3, X_X_3];
        y_pred_TARX(t+2, 3) = predict(mdl, X_combined3);
        if y_pred_TARX(t+2, 3) < 0
            y_pred_TARX(t+2, 3) = 0;
        end
    end

end






%% ------------------------- Plot Predictions ----------------------------

% Plot TARX predictions
function f = plotPredictions(y_true, y_pred, time, titleName)
    % Extract the last 101 points (assuming at least 101 data points)
    y_true_window = y_true(end-100:end);
    y_pred_window = y_pred(end-100:end, :);
    time_window = time(end-100:end);

    % Initialize arrays to store metrics
    RMSE = zeros(1, 3);
    AIC = zeros(1, 3);
    BIC = zeros(1, 3);
    
    f = figure('Units', 'pixels', 'Position', [600, 300, 1000, 600]); % Adjusted for more space
    t = tiledlayout(3, 8, 'TileSpacing', 'compact', 'Padding', 'compact');

    for i = 1:3
        shift = i; % For i-step predictions, predictions lead actual by (i-1)
        
        % Align data:
        % Actual: remove the last 'shift' points, since we can't predict those far ahead
        actual_data = y_true_window(1:end-shift);
        % Predicted: remove the first 'shift' points, as these predictions forecast values further ahead
        pred_data = y_pred_window(shift+1:end, i);
        % Time: must match the length of actual_data
        time_shifted = time_window(1+shift:end);

        % Calculate residuals and metrics
        residuals = actual_data - pred_data;
        RSS = sum(residuals.^2);
        N = length(actual_data);
        k = 1; % Assuming 1 parameter (e.g., single model output per prediction)

        RMSE(i) = sqrt(RSS / N);
        AIC(i) = N * log(RSS / N) + 2 * k;
        BIC(i) = N * log(RSS / N) + k * log(N);
        
        % Display metrics
        fprintf('Step-%d Predictions:\n', i);
        fprintf('  RMSE: %.4f\n', RMSE(i));
        fprintf('  AIC: %.4f\n', AIC(i));
        fprintf('  BIC: %.4f\n\n', BIC(i));
        
        % Plot predictions
        ax1 = nexttile(t, [1,5]); % Specify the span for predictions (6 columns)
        hold on;
        plot(time_shifted, actual_data, 'DisplayName', 'Actual', 'LineWidth', 2, 'Color', 'k');
        plot(time_shifted, pred_data, 'DisplayName', 'Predicted', 'LineWidth', 1.5, 'Color', 'r', 'LineStyle', '--');
        
        if i == 1
            legend('Location','northwest', 'Interpreter','latex');
        end
        title(['Prediction: ', titleName{i}], 'Interpreter', 'latex');
        xlabel('Time', 'Interpreter','latex');
        ylabel('Wind Power','Interpreter','latex');
        xlim([time_shifted(1) time_shifted(end)]);
        grid on;
        box on;
        hold off;
        set(ax1, 'FontSize', 13, 'TickLabelInterpreter', 'latex');
        
        % Plot residuals
        ax2 = nexttile(t, [1, 3]); % Specify the span for residuals (2 columns)
        plot(time_shifted, residuals, 'LineWidth', 1.5, 'Color', 'b');
        title(['Residuals: ', titleName{i}], 'Interpreter', 'latex');
        xlabel('Time', 'Interpreter','latex');
        ylabel('Residual','Interpreter','latex');
        xlim([time_shifted(1) time_shifted(end)]);
        grid on;
        box on;
        set(ax2, 'FontSize', 13, 'TickLabelInterpreter', 'latex');
    end
    
    % Display summary of metrics
    fprintf('Summary of Metrics:\n');
    fprintf('  RMSE: %.4f %.4f %.4f\n', RMSE);
    fprintf('  AIC: %.4f %.4f %.4f\n', AIC);
    fprintf('  BIC: %.4f %.4f %.4f\n', BIC);
end

% Plot predictions dynamically
titles = {'TARX(5,3) - 1-step Predictions', 'TARX(5,3) - 2-step Predictions', 'TARX(5,3) - 3-step Predictions'};
f = plotPredictions(testWindPower, y_pred_TARX, X.t(end-100:end), titles);
exportgraphics(f,'TARX_pred.pdf', 'ContentType', 'vector')

%% Save script to Github

current_file = 'TARX.m';
new_directory = '/home/olivermussmann/Documents/GitHub/02427-AdvancedTimeSeries-CEX4/matlab_code';
copyfile(current_file, new_directory);