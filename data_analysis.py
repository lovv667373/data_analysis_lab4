#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Настройка стилей
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class SpotifyDataAnalyzer:
    """Анализатор данных Spotify для лабораторной работы"""

    def __init__(self):
        self.data = None
        self.results_dir = 'analysis_results'
        os.makedirs(self.results_dir, exist_ok=True)

    def load_data(self):
        """Загрузка датасета"""
        print("=" * 70)
        print("🎵 ЛАБОРАТОРНАЯ РАБОТА: АНАЛИЗ МУЗЫКАЛЬНЫХ ДАННЫХ SPOTIFY")
        print("=" * 70)
        
        try:
            self.data = pd.read_csv('spotify_tracks.csv')
            print(f"✅ Датасет загружен: {len(self.data):,} записей")
            
            # Автоматическое переименование колонок для удобства
            self.rename_columns()
            
            self.show_dataset_info()
            return True
            
        except FileNotFoundError:
            print("❌ Файл spotify_tracks.csv не найден")
            print("📥 Скачиваю датасет...")
            return self.download_dataset()
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            return False