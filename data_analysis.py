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
    
    def create_sample_data(self):
        """Создание демонстрационных данных"""
        np.random.seed(42)
        n_samples = 2000
        
        data = {
            'popularity': np.random.randint(10, 100, n_samples),
            'danceability': np.random.uniform(0.2, 0.95, n_samples),
            'energy': np.random.uniform(0.3, 0.98, n_samples),
            'valence': np.random.uniform(0.1, 0.9, n_samples),
            'tempo': np.random.uniform(60, 180, n_samples),
            'genre': np.random.choice(['pop', 'rock', 'hip-hop', 'edm', 'jazz'], n_samples),
            'duration_ms': np.random.randint(120000, 360000, n_samples)
        }
        
        self.data = pd.DataFrame(data)
        print(f"✅ Создан демонстрационный датасет: {len(self.data):,} записей")
        return True
    
    def show_dataset_info(self):
        """Показать информацию о датасете"""
        print("\n📊 ИНФОРМАЦИЯ О ДАТАСЕТЕ:")
        print(f"• Записей: {len(self.data):,}")
        print(f"• Колонок: {len(self.data.columns)}")
        print(f"• Колонки: {', '.join(self.data.columns[:10])}{'...' if len(self.data.columns) > 10 else ''}")
        
        print("\n🔢 ОСНОВНЫЕ СТАТИСТИКИ:")
        
        # Выбираем только числовые колонки для статистики
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_df = self.data[numeric_cols].describe()
            print(stats_df)

    def prepare_data(self):
        """Подготовка данных"""
        print("\n" + "=" * 70)
        print("🔧 ПОДГОТОВКА ДАННЫХ ДЛЯ АНАЛИЗА")
        print("=" * 70)
        
        # 1. Проверка пропущенных значений
        print("\n1. АНАЛИЗ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ:")
        missing = self.data.isnull().sum()
        missing = missing[missing > 0]
        
        if len(missing) > 0:
            print(f"Найдено пропусков: {len(missing)} колонок")
            for col, count in missing.items():
                percent = (count / len(self.data)) * 100
                print(f"  • {col}: {count} ({percent:.1f}%)")
            
            # Заполняем пропуски
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in missing:
                    self.data[col] = self.data[col].fillna(self.data[col].median())
            
            categorical_cols = self.data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col in missing:
                    self.data[col] = self.data[col].fillna('Unknown')
            
            print("✅ Пропущенные значения обработаны")
        else:
            print("✅ Пропущенных значений нет")
        
        # 2. Создание новых признаков
        print("\n2. СОЗДАНИЕ НОВЫХ ПРИЗНАКОВ:")
        
        # Проверяем наличие необходимых колонок
        created_features = []
        
        # Создаем длительность в минутах
        if 'duration_ms' in self.data.columns:
            self.data['duration_min'] = self.data['duration_ms'] / 60000
            created_features.append('duration_min')
            print(f"  ✓ Создано: duration_min (длительность в минутах)")
        
        # Создаем категорию популярности
        if 'popularity' in self.data.columns:
            try:
                # Используем qcut только если есть достаточно уникальных значений
                unique_vals = self.data['popularity'].nunique()
                if unique_vals >= 3:
                    self.data['popularity_category'] = pd.qcut(
                        self.data['popularity'],
                        q=3,
                        labels=['Низкая', 'Средняя', 'Высокая']
                    )
                    created_features.append('popularity_category')
                    print(f"  ✓ Создано: popularity_category (3 категории)")
                else:
                    # Альтернатива: использовать cut
                    self.data['popularity_category'] = pd.cut(
                        self.data['popularity'],
                        bins=3,
                        labels=['Низкая', 'Средняя', 'Высокая']
                    )
                    created_features.append('popularity_category')
                    print(f"  ✓ Создано: popularity_category (3 интервала)")
            except Exception as e:
                print(f"  ⚠️ Не удалось создать popularity_category: {e}")
        
        # Создаем категорию танцевальности
        if 'danceability' in self.data.columns:
            try:
                self.data['danceability_category'] = pd.cut(
                    self.data['danceability'],
                    bins=[0, 0.4, 0.7, 1],
                    labels=['Низкая', 'Средняя', 'Высокая']
                )
                created_features.append('danceability_category')
                print(f"  ✓ Создано: danceability_category")
            except Exception as e:
                print(f"  ⚠️ Не удалось создать danceability_category: {e}")
        
        # Создаем категорию энергичности
        if 'energy' in self.data.columns:
            conditions = [
                (self.data['energy'] < 0.3),
                (self.data['energy'] >= 0.3) & (self.data['energy'] < 0.7),
                (self.data['energy'] >= 0.7)
            ]
            choices = ['Низкая', 'Средняя', 'Высокая']
            self.data['energy_category'] = np.select(conditions, choices, default='Средняя')
            created_features.append('energy_category')
            print(f"  ✓ Создано: energy_category")
        
        print(f"\n✅ Создано признаков: {len(created_features)}")
        if created_features:
            print(f"Новые признаки: {', '.join(created_features)}")
        
        print("\n🎯 ДАННЫЕ ГОТОВЫ К АНАЛИЗУ")
    