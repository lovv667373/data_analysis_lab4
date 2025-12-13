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
        
    def rename_columns(self):
        """Переименование колонок для удобства"""
        rename_dict = {}
        
        # Автоматически ищем и переименовываем популярные колонки
        column_mapping = {
            'track_popularity': 'popularity',
            'track_name': 'name',
            'track_artist': 'artist', 
            'track_album_name': 'album',
            'duration_ms': 'duration_ms',
            'playlist_genre': 'genre'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in self.data.columns:
                rename_dict[old_name] = new_name
        
        if rename_dict:
            self.data = self.data.rename(columns=rename_dict)
            print(f"📝 Переименовано колонок: {len(rename_dict)}")

    def download_dataset(self):
        """Скачивание датасета если нет локально"""
        try:
            url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"
            self.data = pd.read_csv(url)
            self.data.to_csv('spotify_tracks.csv', index=False)
            print(f"✅ Датасет скачан и сохранен: {len(self.data):,} записей")
            
            self.rename_columns()
            self.show_dataset_info()
            return True
            
        except Exception as e:
            print(f"❌ Не удалось скачать датасет: {e}")
            print("Создаю демонстрационный датасет...")
            return self.create_sample_data()
    