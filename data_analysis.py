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
    
    def visualize_data(self):
        """Визуализация данных"""
        print("\n" + "=" * 70)
        print("📊 ВИЗУАЛИЗАЦИЯ ДАННЫХ С ПОМОЩЬЮ MATPLOTLIB")
        print("=" * 70)
        
        # Создаем несколько графиков
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('АНАЛИЗ МУЗЫКАЛЬНЫХ ДАННЫХ SPOTIFY - ЛАБОРАТОРНАЯ РАБОТА', 
                    fontsize=16, fontweight='bold')
        
        try:
            # 1. Распределение популярности
            if 'popularity' in self.data.columns:
                axes[0, 0].hist(self.data['popularity'], bins=30, 
                               edgecolor='black', alpha=0.7, color='skyblue')
                axes[0, 0].set_title('Распределение популярности треков', fontsize=12)
                axes[0, 0].set_xlabel('Популярность (0-100)')
                axes[0, 0].set_ylabel('Количество треков')
                axes[0, 0].axvline(self.data['popularity'].mean(), color='red', 
                                 linestyle='--', linewidth=2,
                                 label=f'Среднее: {self.data["popularity"].mean():.1f}')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Распределение танцевальности
            if 'danceability' in self.data.columns:
                axes[0, 1].hist(self.data['danceability'], bins=30,
                               edgecolor='black', alpha=0.7, color='lightgreen')
                axes[0, 1].set_title('Распределение танцевальности', fontsize=12)
                axes[0, 1].set_xlabel('Танцевальность (0-1)')
                axes[0, 1].set_ylabel('Количество')
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Популярность по жанрам
            if 'genre' in self.data.columns and 'popularity' in self.data.columns:
                genre_pop = self.data.groupby('genre')['popularity'].mean().sort_values(ascending=False)
                colors = plt.cm.Set3(np.linspace(0, 1, len(genre_pop)))
                genre_pop.plot(kind='bar', ax=axes[0, 2], color=colors, edgecolor='black')
                axes[0, 2].set_title('Средняя популярность по жанрам', fontsize=12)
                axes[0, 2].set_xlabel('Жанр')
                axes[0, 2].set_ylabel('Средняя популярность')
                axes[0, 2].tick_params(axis='x', rotation=45)
                axes[0, 2].grid(True, alpha=0.3)
            
            # 4. Корреляция: танцевальность vs энергичность
            if all(col in self.data.columns for col in ['danceability', 'energy']):
                scatter = axes[1, 0].scatter(self.data['danceability'], 
                                           self.data['energy'], 
                                           alpha=0.5, s=20,
                                           c=self.data.get('popularity', 50),
                                           cmap='viridis')
                axes[1, 0].set_title('Танцевальность vs Энергичность', fontsize=12)
                axes[1, 0].set_xlabel('Танцевальность')
                axes[1, 0].set_ylabel('Энергичность')
                
                # Добавляем линию регрессии
                if len(self.data) > 1:
                    z = np.polyfit(self.data['danceability'], self.data['energy'], 1)
                    p = np.poly1d(z)
                    axes[1, 0].plot(self.data['danceability'], p(self.data['danceability']),
                                   "r--", alpha=0.8, linewidth=2,
                                   label=f'Линия регрессии')
                    axes[1, 0].legend()
                
                if 'popularity' in self.data.columns:
                    plt.colorbar(scatter, ax=axes[1, 0], label='Популярность')
                axes[1, 0].grid(True, alpha=0.3)
            
            # 5. Boxplot танцевальности по жанрам
            if 'genre' in self.data.columns and 'danceability' in self.data.columns:
                # Берем топ-5 жанров
                top_genres = self.data['genre'].value_counts().head(5).index
                genre_data = [self.data[self.data['genre'] == g]['danceability'] 
                            for g in top_genres]
                
                box = axes[1, 1].boxplot(genre_data, labels=top_genres, patch_artist=True)
                colors = plt.cm.Pastel1(np.linspace(0, 1, len(top_genres)))
                for patch, color in zip(box['boxes'], colors):
                    patch.set_facecolor(color)
                
                axes[1, 1].set_title('Танцевальность по жанрам', fontsize=12)
                axes[1, 1].set_xlabel('Жанр')
                axes[1, 1].set_ylabel('Танцевальность')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Распределение темпа
            if 'tempo' in self.data.columns:
                axes[1, 2].hist(self.data['tempo'], bins=30,
                               edgecolor='black', alpha=0.7, color='orange')
                axes[1, 2].set_title('Распределение темпа (BPM)', fontsize=12)
                axes[1, 2].set_xlabel('Темп (ударов в минуту)')
                axes[1, 2].set_ylabel('Количество')
                axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Сохраняем график
            output_path = f'{self.results_dir}/spotify_analysis.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✅ Графики сохранены: {output_path}")
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Ошибка при создании графиков: {e}")
            plt.close()