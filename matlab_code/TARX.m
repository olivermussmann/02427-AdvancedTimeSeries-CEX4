clc; clear; close all;

%% ------------------------- Collect data ---------------------------------

% Read the CSV file
X = readtable('cex4WindDataInterpolated.csv', 'Delimiter', ',', 'ReadVariableNames', true);
X = rmmissing(X); % Remove rows containing NaN's
X.t = hours(X.t - X.t(1));

% Extract relevant columns
windPower = X.p;   % Measured wind power
windSpeed = X.Ws1; % Forecasted wind speed (exogenous input)
windSpeed2 = X.Ws2;
windSpeed3 = X.Ws3;
windDir = X.Wd1;   % Wind direction (for regimes)

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
    if t < length(testWindPower)
        X_AR_2 = [y_pred_TARX(t, 1), X_AR(1:end-1)];
        X_X_2 = [testWindSpeed2(t), X_X(1:end-1)];
        X_combined2 = [X_AR_2, X_X_2];
        y_pred_TARX(t+1, 2) = predict(mdl, X_combined2);
        if y_pred_TARX(t+1, 2) < 0
            y_pred_TARX(t+1, 2) = 0;
        end
    end

    % Three-step prediction
    if t < length(testWindPower)-1
        X_AR_3 = [y_pred_TARX(t+1, 2), X_AR_2(1:end-1)];
        X_X_3 = [testWindSpeed3(t), X_X_2(1:end-1)];
        X_combined3 = [X_AR_3, X_X_3];
        y_pred_TARX(t+2, 3) = predict(mdl, X_combined3);
        if y_pred_TARX(t+2, 3) < 0
            y_pred_TARX(t+2, 3) = 0;
        end
    end

end


%% ------------------------- Performance Metrics -------------------------

% Calculate residuals
residuals1_TARX = testWindPower(lagOrder+1:end) - y_pred_TARX(lagOrder+1:end, 1);
residuals2_TARX = testWindPower(lagOrder+2:end) - y_pred_TARX(lagOrder+2:end, 2);
residuals3_TARX = testWindPower(lagOrder+3:end) - y_pred_TARX(lagOrder+3:end, 3);


%% Calculate RSS, RMSE, AIC, and BIC (One-step)
RSS1_TARX = sum(residuals1_TARX.^2);
RMSE1_TARX = sqrt(mean(residuals1_TARX.^2));
numParams1_TARX = 3 * (lagOrder + 1); % 3 regimes, (lagOrder + 1) parameters per regime
AIC1_TARX = length(testWindPower) * log(RSS1_TARX / length(testWindPower)) + 2 * numParams1_TARX;
BIC1_TARX = length(testWindPower) * log(RSS1_TARX / length(testWindPower)) + numParams1_TARX * log(length(testWindPower));

% Display metrics
disp('Performance metrics of One-step prediction')
disp(['TARX RMSE: ', num2str(RMSE1_TARX)]);
disp(['TARX AIC: ', num2str(AIC1_TARX)]);
disp(['TARX BIC: ', num2str(BIC1_TARX)]);

%% Calculate RSS, RMSE, AIC, and BIC (Two-step)
RSS2_TARX = sum(residuals2_TARX.^2);
RMSE2_TARX = sqrt(mean(residuals2_TARX.^2));
numParams2_TARX = 3 * (lagOrder + 1); % 3 regimes, (lagOrder + 1) parameters per regime
AIC2_TARX = length(testWindPower) * log(RSS2_TARX / length(testWindPower)) + 2 * numParams2_TARX;
BIC2_TARX = length(testWindPower) * log(RSS2_TARX / length(testWindPower)) + numParams2_TARX * log(length(testWindPower));

% Display metrics
disp('Performance metrics of One-step prediction')
disp(['TARX RMSE: ', num2str(RMSE2_TARX)]);
disp(['TARX AIC: ', num2str(AIC2_TARX)]);
disp(['TARX BIC: ', num2str(BIC2_TARX)]);

%% Calculate RSS, RMSE, AIC, and BIC (Three-step)
RSS3_TARX = sum(residuals3_TARX.^2);
RMSE3_TARX = sqrt(mean(residuals3_TARX.^2));
numParams3_TARX = 3 * (lagOrder + 1); % 3 regimes, (lagOrder + 1) parameters per regime
AIC3_TARX = length(testWindPower) * log(RSS3_TARX / length(testWindPower)) + 2 * numParams3_TARX;
BIC3_TARX = length(testWindPower) * log(RSS3_TARX / length(testWindPower)) + numParams3_TARX * log(length(testWindPower));

% Display metrics
disp('Performance metrics of One-step prediction')
disp(['TARX RMSE: ', num2str(RMSE3_TARX)]);
disp(['TARX AIC: ', num2str(AIC3_TARX)]);
disp(['TARX BIC: ', num2str(BIC3_TARX)]);

%% ------------------------- Plot Predictions ----------------------------

% Plot TARX predictions
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


% Plot predictions dynamically
titles = {'TARX(5,3) - 1-step Predictions', 'TARX(5,3) - 2-step Predictions', 'TARX(5,3) - 3-step Predictions'};
f = plotPredictions(testWindPower, y_pred_TARX, X.t(end-100:end), titles);
exportgraphics(f,'TARX_pred.pdf', 'ContentType', 'vector')

%% Save script to Github

current_file = 'TARX.m';
new_directory = '/home/olivermussmann/Documents/GitHub/02427-AdvancedTimeSeries-CEX4/matlab_code';
copyfile(current_file, new_directory);