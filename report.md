## Лабораторная 1: шифрование изображений
1. Алгоритмы и схемы.
- **Потоковый** (XOR с ключевым потоком): \
  Получаем **keystream** длиной в байты изображения с помощью метода LCG \
  В качестве первого значения берём key + iv \
  В конце LCG проходим весь keystream по модулю 255 \
  С помощью keystream шифруем изображение `cipher = (plaintext) XOR (keystream)`
- **Перестановочный (confusion + diffusion):** \
  Для перестановки выбрал Arnold cat map с 7 итерациями \
  Для каждого канала делаем отдельную перестановку \
  \+ Дополнительно делаем подстановку xor потоком
2. Формирование/хранение IV/nonce. \
  IV формируется так - `iv = os.urandom(16).hex()` \
  Функция os.urandom() возвращает строку случайных байтов размером 16 \
  При каждом шифровании iv формируется заново, сохраняется в META.json
3. Метрики (таблицы/графики) и выводы. \
- **Потоковый метод**\
**Гистограмма**
![encryption_test_histograms.png](results%2Fencryption_test_histograms.png)
**Корреляция соседних пикселей**
![encryption_test_correlation.png](results%2Fencryption_test_correlation.png)
**Энтропия каналов**
![encryption_test_entropy.png](results%2Fencryption_test_entropy.png)
**NPCR / UACI** \
"npcr": 99.60962931315103 \
"uaci": 35.742720560310715 \
**Чувствительность к ключу (avalanche)** 
![encryption_test_avalanche.png](results%2Fencryption_test_avalanche.png)
- **Перестановочный метод**\
**Гистограмма**
![encryption_test_histograms2.png](results%2Fencryption_test_histograms2.png)
**Корреляция соседних пикселей**
![encryption_test_correlation2.png](results%2Fencryption_test_correlation2.png)
**Энтропия каналов**
![encryption_test_entropy2.png](results%2Fencryption_test_entropy2.png)
**NPCR / UACI** \
"npcr": 99.62005615234375 \
"uaci": 32.079162099003014 \
**Чувствительность к ключу (avalanche)**
![encryption_test2_avalanche.png](results%2Fencryption_test2_avalanche.png)