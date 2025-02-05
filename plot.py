import os
import glob
import json
import matplotlib.pyplot as plt
import numpy as np

def load_experiment_data(experiments_dir):
    """
    Загружает данные экспериментов из JSON-файлов в папке experiments_dir,
    имена которых начинаются с восклицательного знака.
    """
    pattern = os.path.join(experiments_dir, '!*.json')
    json_files = glob.glob(pattern)
    
    if not json_files:
        print(f'Не найдено файлов, начинающихся с "!" в папке {experiments_dir}')
    
    experiments = []
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            # Извлекаем имя эксперимента (имя файла без расширения)
            label = os.path.splitext(os.path.basename(file_path))[0]
            experiments.append({'label': label, 'data': data})
        except Exception as e:
            print(f'Ошибка при загрузке файла {file_path}: {e}')
    return experiments

def moving_average(data, window_size):
    """
    Вычисляет скользящее среднее и стандартное отклонение по данным.
    Возвращает два массива: среднее и std, размер которых совпадает с исходным.
    """
    data = np.array(data)
    ma = []
    std = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        window = data[start:i+1]
        ma.append(window.mean())
        std.append(window.std())
    return np.array(ma), np.array(std)

def plot_experiments(experiments):
    """
    Строит 4 подграфика:
      - Верхняя строка: accuracy (обучающая и тестовая) с общим вертикальным масштабом.
      - Нижняя строка: loss (обучающая и тестовая) с общим вертикальным масштабом.
    Для каждого эксперимента строится линия с подписью (label).
    """
    window_size = 390

    # Создаем фигуру с 2 строками и 2 столбцами, 
    # при этом подграфики в каждой строке будут делить ось Y (sharey='row')
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharey='row')
    
    # Заголовки и подписи осей для каждого подграфика
    
    for ax in axs.flat:
        ax.set_xlabel('# full gradient computations')

    axs[0, 0].set_title('Train Accuracy')
    axs[0, 0].set_ylabel('Accuracy')
    
    axs[0, 1].set_title('Test Accuracy')
    axs[0, 1].set_ylabel('Accuracy')
    
    axs[1, 0].set_title('Train Loss')
    axs[1, 0].set_ylabel('Loss')
    
    axs[1, 1].set_title('Test Loss')
    axs[1, 1].set_ylabel('Loss')


    for ax in axs.flat:
        ax.yaxis.set_tick_params(labelleft=True)

    
    # Для каждого эксперимента строим линии на всех подграфиках
    for exp in experiments:
        label = exp['label']
        data = exp['data']
        
        # Проверяем наличие необходимых ключей
        if 'train' in data and 'test' in data:
            # Обучающая выборка
            train_index = data['train'].get('index', [])
            train_accuracy = data['train'].get('accuracy', [])
            train_loss = data['train'].get('loss', [])
            
            # Тестовая выборка
            test_index = data['test'].get('index', [])
            test_accuracy = data['test'].get('accuracy', [])
            test_loss = data['test'].get('loss', [])
            
            # Строим линии с маркерами для лучшей видимости
            axs[0, 1].plot(test_index, test_accuracy, linestyle="-", label=label)
            axs[1, 1].semilogy(test_index, test_loss, linestyle="-", label=label)

            # Вычисление скользящего среднего и стандартного отклонения для тестовых данных
            ma_acc, std_acc = moving_average(train_accuracy, window_size)
            ma_loss, std_loss = moving_average(train_loss, window_size)
            
            # Построение тестовых графиков с errorbar
            acc_line = axs[0, 0].plot(train_index, ma_acc, linestyle="-", label=label)[0]
            acc_color = acc_line.get_color()
            lss_line = axs[1, 0].semilogy(train_index, ma_loss, linestyle="-", label=label)[0]
            lss_color = lss_line.get_color()

            # Заливаем область от (ma_acc - std_acc) до (ma_acc + std_acc)
            axs[0, 0].fill_between(train_index, 
                                np.array(ma_acc) - np.array(std_acc), 
                                np.array(ma_acc) + np.array(std_acc), 
                                color=acc_color,
                                alpha=0.1)  # alpha отвечает за прозрачность
            
            # Для потерь используем semilog, поэтому сначала строим линию, затем заливаем область, а потом устанавливаем логарифмическую ось
            axs[1, 0].fill_between(train_index, 
                                np.array(ma_loss) - np.array(std_loss), 
                                np.array(ma_loss) + np.array(std_loss), 
                                color=lss_color,
                                alpha=0.1)
            axs[1, 0].set_yscale('log')
        else:
            print(f'В данных эксперимента {label} отсутствуют необходимые ключи.')
    
    for ax in axs.flat:
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('experiments_plot.png')
    # plt.show()

if __name__ == '__main__':
    experiments_directory = 'experiments'
    experiments_data = load_experiment_data(experiments_directory)    
    if experiments_data:
        plot_experiments(experiments_data)
