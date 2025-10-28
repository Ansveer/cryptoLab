# metrics.py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import json

class CryptoMetrics:
    def __init__(self):
        self.fig_size = (15, 10)
    
    def calculate_entropy(self, data):
        """Вычисление энтропии Шеннона"""
        if len(data) == 0:
            return 0
        
        # Нормализация данных к целым числам 0-255
        data_int = data.astype(np.uint8)
        
        # Вычисление гистограммы
        hist, _ = np.histogram(data_int, bins=256, range=(0, 256))
        hist = hist[hist > 0]  # Убираем нулевые значения
        
        # Вычисление вероятностей
        probabilities = hist / len(data_int)
        
        # Вычисление энтропии
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def calculate_correlation(self, img, direction='horizontal'):
        """Вычисление корреляции соседних пикселей"""
        if len(img.shape) == 3:
            img = img.mean(axis=2)  # Преобразуем в grayscale
        
        h, w = img.shape
        
        if direction == 'horizontal':
            x = img[:, :-1].flatten()
            y = img[:, 1:].flatten()
        elif direction == 'vertical':
            x = img[:-1, :].flatten()
            y = img[1:, :].flatten()
        elif direction == 'diagonal':
            x = img[:-1, :-1].flatten()
            y = img[1:, 1:].flatten()
        else:
            raise ValueError("Direction must be 'horizontal', 'vertical', or 'diagonal'")
        
        correlation = np.corrcoef(x, y)[0, 1]
        return correlation if not np.isnan(correlation) else 0
    
    def calculate_npcr_uaci(self, img1, img2):
        """Вычисление NPCR и UACI между двумя изображениями"""
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same shape")
        
        # Преобразуем в uint8 для точных вычислений
        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)
        
        # NPCR (Number of Pixels Change Rate)
        diff_pixels = np.sum(img1 != img2)
        total_pixels = img1.size
        npcr = (diff_pixels / total_pixels) * 100
        
        # UACI (Unified Average Changing Intensity)
        diff = np.abs(img1.astype(float) - img2.astype(float))
        uaci = (np.sum(diff) / (255 * img1.size)) * 100
        
        return npcr, uaci

    def calculate_bit_change_rate(self, original_bits, modified_bits):
        """
        Вычисляет процент измененных бит между двумя массивами
        """
        total_bits = len(original_bits) * 8  # каждый элемент - 8 бит
        changed_bits = 0
        
        for orig_byte, mod_byte in zip(original_bits, modified_bits):
            # XOR для выявления различий в битах
            xor_result = orig_byte ^ mod_byte
            # Подсчитываем количество установленных битов (измененных)
            changed_bits += bin(xor_result).count('1')
        
        return (changed_bits / total_bits) * 100

    def key_avalanche_test(self, original_img, key, algo, iterations=10):
        """
        Тестирует лавинный эффект при изменении ключа
        
        Args:
            original_img: оригинальное изображение (numpy array)
            key: исходный ключ
            algo: алгоритм шифрования ("stream" или "perm-mix")
            iterations: количество тестовых битов
        """
        print(f"    Testing avalanche effect for {algo} algorithm...")
        
        height, width = original_img.shape[0], original_img.shape[1]
        if len(original_img.shape) == 3:
            channels = original_img.shape[2]
            flat_img = original_img.flatten()
        else:
            channels = 1
            flat_img = original_img.flatten()
        
        # Генерируем IV
        iv = os.urandom(16).hex()
        
        # Импортируем функции шифрования
        from cryptopic import keystreamGenLCG, xorStream, arnoldCatEncrypt
        
        # Шифруем с оригинальным ключом
        original_key = int(str(key) + str(int(iv, 16)))
        keystream_original = keystreamGenLCG(len(flat_img), original_key)
        
        if algo == "stream":
            encrypted_original = xorStream(flat_img, keystream_original, len(flat_img))
            encrypted_original_img = np.array(encrypted_original).reshape(original_img.shape)
        elif algo == "perm-mix":
            encrypted_arr = original_img.copy()
            iterations_arnold = 7
            for j in range(channels):
                channel = encrypted_arr[:, :, j]
                encrypted_channel = arnoldCatEncrypt(channel.flatten(), width, height, iterations_arnold, original_key)
                encrypted_arr[:, :, j] = encrypted_channel.reshape((height, width))
            encrypted_original = xorStream(encrypted_arr.flatten(), keystream_original, len(flat_img))
            encrypted_original_img = np.array(encrypted_original).reshape(original_img.shape)
        
        avalanche_results = []
        avalanche_npcr_results = []
        avalanche_uaci_results = []
        bit_positions = []
        
        # Тестируем изменение каждого бита в ключе
        for bit_pos in range(min(64, iterations)):  # тестируем первые 64 бита или меньше
            # Изменяем один бит в ключе
            modified_key = original_key ^ (1 << bit_pos)
            
            # Шифруем с измененным ключом
            keystream_modified = keystreamGenLCG(len(flat_img), modified_key)
            
            if algo == "stream":
                encrypted_modified = xorStream(flat_img, keystream_modified, len(flat_img))
                encrypted_modified_img = np.array(encrypted_modified).reshape(original_img.shape)
            elif algo == "perm-mix":
                encrypted_arr_modified = original_img.copy()
                for j in range(channels):
                    channel = encrypted_arr_modified[:, :, j]
                    encrypted_channel = arnoldCatEncrypt(channel.flatten(), width, height, iterations_arnold, modified_key)
                    encrypted_arr_modified[:, :, j] = encrypted_channel.reshape((height, width))
                encrypted_modified = xorStream(encrypted_arr_modified.flatten(), keystream_modified, len(flat_img))
                encrypted_modified_img = np.array(encrypted_modified).reshape(original_img.shape)
            
            # Вычисляем процент измененных бит (битовый уровень)
            bit_change_rate = self.calculate_bit_change_rate(encrypted_original, encrypted_modified)
            avalanche_results.append(bit_change_rate)
            
            # Вычисляем NPCR и UACI (пиксельный уровень)
            npcr, uaci = self.calculate_npcr_uaci(encrypted_original_img, encrypted_modified_img)
            avalanche_npcr_results.append(npcr)
            avalanche_uaci_results.append(uaci)
            
            bit_positions.append(bit_pos)
        
        return (avalanche_results, avalanche_npcr_results, avalanche_uaci_results, 
                bit_positions, encrypted_original_img)

    def plot_avalanche_analysis(self, avalanche_results, avalanche_npcr_results, 
                              avalanche_uaci_results, bit_positions, algo, filename):
        """Визуализация анализа лавинного эффекта"""
        # Статистика для битовых изменений
        avg_change = np.mean(avalanche_results)
        std_change = np.std(avalanche_results)
        min_change = np.min(avalanche_results)
        max_change = np.max(avalanche_results)
        ideal_value = 50.0
        deviation = abs(avg_change - ideal_value)
        
        # Статистика для NPCR
        avg_npcr = np.mean(avalanche_npcr_results)
        std_npcr = np.std(avalanche_npcr_results)
        ideal_npcr = 99.61  # Идеальное значение для NPCR
        
        # Статистика для UACI
        avg_uaci = np.mean(avalanche_uaci_results)
        std_uaci = np.std(avalanche_uaci_results)
        ideal_uaci = 33.46  # Идеальное значение для UACI
        
        # Создаем графики
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. График битовых изменений
        ax1.plot(bit_positions, avalanche_results, 'bo-', alpha=0.7, linewidth=2, markersize=4, label='Bit Change Rate')
        ax1.axhline(y=50, color='red', linestyle='--', label='Идеал (50%)', alpha=0.7)
        ax1.axhline(y=avg_change, color='green', linestyle='--', label=f'Среднее ({avg_change:.1f}%)', alpha=0.7)
        ax1.set_xlabel('Позиция измененного бита в ключе')
        ax1.set_ylabel('Процент измененных бит в шифре (%)')
        ax1.set_title(f'Лавинный эффект (битовый уровень) - {algo}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. График NPCR изменений
        ax2.plot(bit_positions, avalanche_npcr_results, 'go-', alpha=0.7, linewidth=2, markersize=4, label='NPCR')
        ax2.axhline(y=ideal_npcr, color='red', linestyle='--', label=f'Идеал NPCR ({ideal_npcr}%)', alpha=0.7)
        ax2.axhline(y=avg_npcr, color='blue', linestyle='--', label=f'Среднее NPCR ({avg_npcr:.1f}%)', alpha=0.7)
        ax2.set_xlabel('Позиция измененного бита в ключе')
        ax2.set_ylabel('NPCR (%)')
        ax2.set_title(f'Чувствительность ключа (NPCR) - {algo}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. График UACI изменений
        ax3.plot(bit_positions, avalanche_uaci_results, 'mo-', alpha=0.7, linewidth=2, markersize=4, label='UACI')
        ax3.axhline(y=ideal_uaci, color='red', linestyle='--', label=f'Идеал UACI ({ideal_uaci}%)', alpha=0.7)
        ax3.axhline(y=avg_uaci, color='blue', linestyle='--', label=f'Среднее UACI ({avg_uaci:.1f}%)', alpha=0.7)
        ax3.set_xlabel('Позиция измененного бита в ключе')
        ax3.set_ylabel('UACI (%)')
        ax3.set_title(f'Чувствительность ключа (UACI) - {algo}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Сводная статистика
        metrics = ['Bit Change', 'NPCR', 'UACI']
        avg_values = [avg_change, avg_npcr, avg_uaci]
        ideal_values = [ideal_value, ideal_npcr, ideal_uaci]
        deviations = [deviation, abs(avg_npcr - ideal_npcr), abs(avg_uaci - ideal_uaci)]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax4.bar(x - width/2, avg_values, width, label='Среднее', alpha=0.7)
        ax4.bar(x + width/2, ideal_values, width, label='Идеал', alpha=0.7)
        
        # Добавляем значения отклонений
        for i, (avg, ideal, dev) in enumerate(zip(avg_values, ideal_values, deviations)):
            ax4.text(i - width/2, avg + 1, f'{avg:.1f}%', ha='center', va='bottom', fontsize=9)
            ax4.text(i + width/2, ideal + 1, f'{ideal:.1f}%', ha='center', va='bottom', fontsize=9)
            ax4.text(i, min(avg, ideal) - 5, f'Δ={dev:.1f}%', ha='center', va='top', fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        ax4.set_xlabel('Метрики')
        ax4.set_ylabel('Процент (%)')
        ax4.set_title('Сравнение метрик чувствительности ключа')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'../results/{filename}_avalanche.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'bit_level_analysis': {
                'average_change_rate': float(avg_change),
                'standard_deviation': float(std_change),
                'min_change_rate': float(min_change),
                'max_change_rate': float(max_change),
                'deviation_from_ideal': float(deviation),
                'bit_change_rates': [float(x) for x in avalanche_results],
            },
            'pixel_level_analysis': {
                'npcr': {
                    'average': float(avg_npcr),
                    'standard_deviation': float(std_npcr),
                    'ideal_value': float(ideal_npcr),
                    'deviation_from_ideal': float(abs(avg_npcr - ideal_npcr)),
                    'values': [float(x) for x in avalanche_npcr_results]
                },
                'uaci': {
                    'average': float(avg_uaci),
                    'standard_deviation': float(std_uaci),
                    'ideal_value': float(ideal_uaci),
                    'deviation_from_ideal': float(abs(avg_uaci - ideal_uaci)),
                    'values': [float(x) for x in avalanche_uaci_results]
                }
            },
            'tested_bit_positions': bit_positions,
            'overall_assessment': {
                'bit_change_quality': 'Excellent' if deviation < 5 else 'Good' if deviation < 10 else 'Fair' if deviation < 20 else 'Poor',
                'npcr_quality': 'Excellent' if abs(avg_npcr - ideal_npcr) < 0.1 else 'Good' if abs(avg_npcr - ideal_npcr) < 0.5 else 'Fair' if abs(avg_npcr - ideal_npcr) < 1 else 'Poor',
                'uaci_quality': 'Excellent' if abs(avg_uaci - ideal_uaci) < 0.1 else 'Good' if abs(avg_uaci - ideal_uaci) < 0.5 else 'Fair' if abs(avg_uaci - ideal_uaci) < 1 else 'Poor'
            }
        }
    
    def plot_histograms(self, original_img, encrypted_img, filename):
        """Гистограммы каналов до и после шифрования"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        channels = ['Red', 'Green', 'Blue']
        
        for i, color in enumerate(['red', 'green', 'blue']):
            # Оригинальное изображение
            if len(original_img.shape) == 3:
                orig_channel = original_img[:, :, i].flatten()
            else:
                orig_channel = original_img.flatten()
            
            axes[0, i].hist(orig_channel, bins=256, color=color, alpha=0.7)
            axes[0, i].set_title(f'Original {channels[i]} Channel')
            axes[0, i].set_xlim(0, 255)
            
            # Зашифрованное изображение
            if len(encrypted_img.shape) == 3:
                enc_channel = encrypted_img[:, :, i].flatten()
            else:
                enc_channel = encrypted_img.flatten()
            
            axes[1, i].hist(enc_channel, bins=256, color=color, alpha=0.7)
            axes[1, i].set_title(f'Encrypted {channels[i]} Channel')
            axes[1, i].set_xlim(0, 255)
        
        plt.tight_layout()
        plt.savefig(f'../results/{filename}_histograms.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_correlation_analysis(self, original_img, encrypted_img, filename):
        """Анализ корреляции соседних пикселей"""
        directions = ['horizontal', 'vertical', 'diagonal']
        orig_correlations = []
        enc_correlations = []
        
        for direction in directions:
            orig_corr = self.calculate_correlation(original_img, direction)
            enc_corr = self.calculate_correlation(encrypted_img, direction)
            orig_correlations.append(orig_corr)
            enc_correlations.append(enc_corr)
        
        # График корреляций
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(directions))
        width = 0.35
        
        ax.bar(x - width/2, orig_correlations, width, label='Original', alpha=0.7)
        ax.bar(x + width/2, enc_correlations, width, label='Encrypted', alpha=0.7)
        
        ax.set_xlabel('Direction')
        ax.set_ylabel('Correlation Coefficient')
        ax.set_title('Pixel Correlation Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(['Horizontal', 'Vertical', 'Diagonal'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'../results/{filename}_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return dict(zip(directions, orig_correlations)), dict(zip(directions, enc_correlations))
    
    def plot_entropy_analysis(self, original_img, encrypted_img, filename):
        """Анализ энтропии каналов"""
        if len(original_img.shape) == 3:
            channels = ['Red', 'Green', 'Blue']
            orig_entropies = []
            enc_entropies = []
            
            for i in range(3):
                orig_entropy = self.calculate_entropy(original_img[:, :, i])
                enc_entropy = self.calculate_entropy(encrypted_img[:, :, i])
                orig_entropies.append(orig_entropy)
                enc_entropies.append(enc_entropy)
            
            # Общая энтропия
            orig_total_entropy = self.calculate_entropy(original_img.flatten())
            enc_total_entropy = self.calculate_entropy(encrypted_img.flatten())
        else:
            channels = ['Grayscale']
            orig_entropies = [self.calculate_entropy(original_img)]
            enc_entropies = [self.calculate_entropy(encrypted_img)]
            orig_total_entropy = orig_entropies[0]
            enc_total_entropy = enc_entropies[0]
        
        # График энтропии
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Энтропия по каналам
        x = np.arange(len(channels))
        width = 0.35
        
        ax1.bar(x - width/2, orig_entropies, width, label='Original', alpha=0.7)
        ax1.bar(x + width/2, enc_entropies, width, label='Encrypted', alpha=0.7)
        ax1.axhline(y=8, color='red', linestyle='--', label='Ideal (8 bits)')
        
        ax1.set_xlabel('Channels')
        ax1.set_ylabel('Entropy (bits)')
        ax1.set_title('Channel Entropy')
        ax1.set_xticks(x)
        ax1.set_xticklabels(channels)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Общая энтропия
        categories = ['Original', 'Encrypted', 'Ideal']
        values = [orig_total_entropy, enc_total_entropy, 8.0]
        colors = ['blue', 'orange', 'red']
        
        ax2.bar(categories, values, color=colors, alpha=0.7)
        ax2.set_ylabel('Entropy (bits)')
        ax2.set_title('Total Entropy Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Добавляем значения на столбцы
        for i, v in enumerate(values):
            ax2.text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'../results/{filename}_entropy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return orig_entropies, enc_entropies, orig_total_entropy, enc_total_entropy
    
    def generate_comprehensive_report(self, original_path, encrypted_path, key, algo, output_filename):
        """Генерация полного отчета со всеми метриками"""
        # Загрузка изображений
        original_img = np.asarray(Image.open(f"../imgs/{original_path}"))
        encrypted_img = np.asarray(Image.open(f"../imgs/{encrypted_path}"))
        
        print(f"Generating metrics report for {algo} algorithm...")
        
        # 1. Гистограммы
        print("1. Generating histograms...")
        self.plot_histograms(original_img, encrypted_img, output_filename)
        
        # 2. Корреляционный анализ
        print("2. Analyzing pixel correlations...")
        orig_corr, enc_corr = self.plot_correlation_analysis(original_img, encrypted_img, output_filename)
        
        # 3. Анализ энтропии
        print("3. Calculating entropy...")
        orig_entropies, enc_entropies, orig_total_entropy, enc_total_entropy = self.plot_entropy_analysis(
            original_img, encrypted_img, output_filename
        )
        
        # 4. NPCR/UACI анализ между оригиналом и шифром
        print("4. Calculating NPCR/UACI between original and encrypted...")
        npcr_orig_enc, uaci_orig_enc = self.calculate_npcr_uaci(original_img, encrypted_img)
        
        # 5. Avalanche effect с NPCR/UACI анализом
        print("5. Testing key avalanche effect with NPCR/UACI...")
        (avalanche_results, avalanche_npcr_results, avalanche_uaci_results, 
         bit_positions, reference_encrypted) = self.key_avalanche_test(original_img, key, algo, iterations=20)
        
        avalanche_analysis = self.plot_avalanche_analysis(
            avalanche_results, avalanche_npcr_results, avalanche_uaci_results, 
            bit_positions, algo, output_filename
        )
        
        # Создаем полный отчет
        report = {
            'algorithm': algo,
            'key_used': str(key),
            'image_size': original_img.shape,
            'correlation_analysis': {
                'original': orig_corr,
                'encrypted': enc_corr
            },
            'entropy_analysis': {
                'original_channels': [float(e) for e in orig_entropies],
                'encrypted_channels': [float(e) for e in enc_entropies],
                'original_total': float(orig_total_entropy),
                'encrypted_total': float(enc_total_entropy)
            },
            'npcr_uaci_analysis': {
                'original_vs_encrypted': {
                    'npcr': float(npcr_orig_enc),
                    'uaci': float(uaci_orig_enc)
                }
            },
            'key_sensitivity_analysis': avalanche_analysis,
            'security_metrics': {
                'ideal_entropy': 8.0,
                'ideal_avalanche': 50.0,
                'ideal_npcr': 99.61,
                'ideal_uaci': 33.46,
                'ideal_correlation': 0.0
            }
        }
        
        # Сохраняем отчет
        with open(f'../results/{output_filename}_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Создаем сводный график
        self.create_summary_plot(report, output_filename)
        
        print(f"Metrics report generated: ../results/{output_filename}_report.json")
        return report
    
    def create_summary_plot(self, report, filename):
        """Создание сводного графика с основными метриками"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Корреляция
        directions = list(report['correlation_analysis']['original'].keys())
        orig_corr = [report['correlation_analysis']['original'][d] for d in directions]
        enc_corr = [report['correlation_analysis']['encrypted'][d] for d in directions]
        
        x = np.arange(len(directions))
        ax1.bar(x - 0.2, orig_corr, 0.4, label='Original', alpha=0.7)
        ax1.bar(x + 0.2, enc_corr, 0.4, label='Encrypted', alpha=0.7)
        ax1.set_title('Pixel Correlation')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['H', 'V', 'D'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Энтропия
        entropy_data = [
            report['entropy_analysis']['original_total'],
            report['entropy_analysis']['encrypted_total'],
            8.0
        ]
        entropy_labels = ['Original', 'Encrypted', 'Ideal']
        colors = ['blue', 'orange', 'green']
        ax2.bar(entropy_labels, entropy_data, color=colors, alpha=0.7)
        ax2.set_title('Information Entropy')
        ax2.set_ylim(0, 8.5)
        ax2.grid(True, alpha=0.3)
        
        # 3. NPCR/UACI между оригиналом и шифром
        metrics_orig_enc = ['NPCR', 'UACI']
        values_orig_enc = [
            report['npcr_uaci_analysis']['original_vs_encrypted']['npcr'],
            report['npcr_uaci_analysis']['original_vs_encrypted']['uaci']
        ]
        ideal_values_orig_enc = [99.61, 33.46]
        
        x_npcr_uaci = np.arange(len(metrics_orig_enc))
        width = 0.35
        
        ax3.bar(x_npcr_uaci - width/2, values_orig_enc, width, label='Actual', alpha=0.7)
        ax3.bar(x_npcr_uaci + width/2, ideal_values_orig_enc, width, label='Ideal', alpha=0.7)
        ax3.set_title('NPCR/UACI (Original vs Encrypted)')
        ax3.set_xticks(x_npcr_uaci)
        ax3.set_xticklabels(metrics_orig_enc)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Добавляем значения
        for i, (actual, ideal) in enumerate(zip(values_orig_enc, ideal_values_orig_enc)):
            ax3.text(i - width/2, actual + 1, f'{actual:.1f}%', ha='center', va='bottom')
            ax3.text(i + width/2, ideal + 1, f'{ideal:.1f}%', ha='center', va='bottom')
        
        # 4. Чувствительность ключа (сводка)
        key_sens = report['key_sensitivity_analysis']
        metrics_keysens = ['Bit Change', 'NPCR', 'UACI']
        actual_values = [
            key_sens['bit_level_analysis']['average_change_rate'],
            key_sens['pixel_level_analysis']['npcr']['average'],
            key_sens['pixel_level_analysis']['uaci']['average']
        ]
        ideal_values = [50.0, 99.61, 33.46]
        
        x_keysens = np.arange(len(metrics_keysens))
        
        ax4.bar(x_keysens - width/2, actual_values, width, label='Actual', alpha=0.7, color='cyan')
        ax4.bar(x_keysens + width/2, ideal_values, width, label='Ideal', alpha=0.7, color='magenta')
        ax4.set_title('Key Sensitivity Summary')
        ax4.set_xticks(x_keysens)
        ax4.set_xticklabels(metrics_keysens)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Добавляем значения и оценки качества
        qualities = [
            key_sens['overall_assessment']['bit_change_quality'],
            key_sens['overall_assessment']['npcr_quality'],
            key_sens['overall_assessment']['uaci_quality']
        ]
        
        for i, (actual, ideal, quality) in enumerate(zip(actual_values, ideal_values, qualities)):
            ax4.text(i - width/2, actual + 1, f'{actual:.1f}%', ha='center', va='bottom', fontsize=8)
            ax4.text(i + width/2, ideal + 1, f'{ideal:.1f}%', ha='center', va='bottom', fontsize=8)
            ax4.text(i, min(actual, ideal) - 3, quality, ha='center', va='top', fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen" if quality in ['Excellent', 'Good'] else "yellow", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f'../results/{filename}_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

# Функция для использования в основном скрипте
def generate_metrics(original_image, encrypted_image, key, algorithm, test_name="metrics"):
    """Основная функция для генерации метрик"""
    metrics = CryptoMetrics()
    report = metrics.generate_comprehensive_report(
        original_image, encrypted_image, key, algorithm, test_name
    )
    return report