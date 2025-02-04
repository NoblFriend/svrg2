import os
import glob
import json
import matplotlib.pyplot as plt

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

def plot_experiments(experiments):
    """
    Строит 4 подграфика:
      - Верхняя строка: accuracy (обучающая и тестовая) с общим вертикальным масштабом.
      - Нижняя строка: loss (обучающая и тестовая) с общим вертикальным масштабом.
    Для каждого эксперимента строится линия с подписью (label).
    """
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
            axs[0, 0].plot(train_index, train_accuracy, linestyle="-", label=label)
            axs[0, 1].plot(test_index, test_accuracy, linestyle="-", label=label)
            axs[1, 0].semilogy(train_index, train_loss, linestyle="-", label=label)
            axs[1, 1].semilogy(test_index, test_loss, linestyle="-", label=label)
        else:
            print(f'В данных эксперимента {label} отсутствуют необходимые ключи.')
    
    # Добавляем сетку и легенды для всех подграфиков
    for ax in axs.flat:
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('experiments_plot.png')
    # plt.show()

if __name__ == '__main__':
    # Укажите путь к папке с экспериментами
    experiments_directory = 'experiments'
    
    # Загружаем данные экспериментов
    experiments_data = load_experiment_data(experiments_directory)
    
    # Строим графики, если найдены данные
    if experiments_data:
        plot_experiments(experiments_data)
