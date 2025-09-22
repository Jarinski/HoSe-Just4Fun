#!/usr/bin/env python3
# smart_football_llm.py - Intelligentes LLM das selbst nach Spielern sucht

import pandas as pd
import anthropic
import json
import re
import streamlit as st
import os
from Statistik import SmartFootballAnalyzer

API_KEY = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

class SmartFootballAnalyzer:
    def __init__(self):
        self.df = None
        self.client = None
        self.all_players = []
        
    def load_data(self):
        """Daten laden und alle Spielernamen extrahieren"""
        try:
            self.df = pd.read_csv('events_long_parsed.csv')
            self.all_players = sorted(self.df['player_name'].dropna().unique().tolist())
            print(f"✅ Daten geladen: {len(self.df)} Zeilen, {len(self.all_players)} Spieler")
            return True
        except FileNotFoundError:
            print("❌ events_long_parsed.csv nicht gefunden")
            return False
    
    def setup_claude(self):
        """Claude API Setup"""
        try:
            self.client = anthropic.Anthropic(api_key=API_KEY)
            response = self.client.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=10,
                messages=[{"role": "user", "content": "Test"}]
            )
            print("✅ Claude API funktioniert!")
            return True
        except Exception as e:
            print(f"❌ Claude API Fehler: {e}")
            return False
    
    def find_players_in_question(self, question):
        """Findet automatisch alle Spielernamen in der Frage"""
        question_lower = question.lower()
        found_players = []
        
        # Direkte Namenssuche
        for player in self.all_players:
            player_words = player.lower().split()
            
            # Vollständiger Name
            if player.lower() in question_lower:
                found_players.append(player)
                continue
                
            # Teilname (Vor- oder Nachname)
            for word in player_words:
                if len(word) > 3 and word in question_lower:
                    if player not in found_players:
                        found_players.append(player)
                        break
        
        return found_players
    
    def get_comprehensive_time_analysis(self):
        """Umfassende Analyse aller Zeitintervalle"""
        import ast
        
        # Definiere verschiedene Zeitintervalle
        time_intervals = {
            '1-10': (1, 10),
            '11-20': (11, 20), 
            '21-30': (21, 30),
            '31-40': (31, 40),
            '41-45': (41, 45),
            '46-55': (46, 55),
            '56-65': (56, 65),
            '66-75': (66, 75),
            '76-85': (76, 85),
            '86-90': (86, 90),
            '90+': (91, 120)  # Nachspielzeit
        }
        
        interval_goals = {interval: {} for interval in time_intervals.keys()}
        
        for _, game in self.df.iterrows():
            if game['goals'] > 0 and pd.notna(game['goals_minutes']) and game['goals_minutes'] != '[]':
                try:
                    minutes = ast.literal_eval(game['goals_minutes'])
                    if not isinstance(minutes, list):
                        continue
                    
                    player_name = game['player_name']
                    
                    for minute in minutes:
                        if isinstance(minute, (int, float)):
                            minute = int(minute)
                            
                            # Finde passendes Intervall
                            for interval_name, (start, end) in time_intervals.items():
                                if start <= minute <= end:
                                    if player_name not in interval_goals[interval_name]:
                                        interval_goals[interval_name][player_name] = 0
                                    interval_goals[interval_name][player_name] += 1
                                    break
                                    
                except (ValueError, SyntaxError, TypeError):
                    continue
        
        # Sortiere für jedes Intervall - TYPE-SAFE
        sorted_intervals = {}
        for interval, goals in interval_goals.items():
            # Filtere nur valide Einträge (string keys, numeric values)
            valid_goals = []
            for player, goal_count in goals.items():
                if isinstance(player, str) and isinstance(goal_count, (int, float)) and not pd.isna(goal_count):
                    valid_goals.append((player, int(goal_count)))
            
            sorted_intervals[interval] = sorted(valid_goals, key=lambda x: x[1], reverse=True)[:10]
        
        # Gesamtstatistiken
        total_stats = {}
        for interval, players in sorted_intervals.items():
            total_goals = sum(goals for _, goals in players)
            total_stats[interval] = {
                'total_goals': total_goals,
                'top_scorer': players[0] if players else None,
                'active_players': len([p for p, g in players if g > 0])
            }
        
        return {
            "time_interval_analysis": sorted_intervals,
            "interval_statistics": total_stats,
            "analysis_intervals": time_intervals
        }

    def get_late_game_scorers(self):
        """Findet wer in den letzten Minuten am häufigsten trifft"""
        import ast
        
        late_game_goals = {}  # {player: goals_count}
        
        for _, game in self.df.iterrows():
            if game['goals'] > 0 and pd.notna(game['goals_minutes']) and game['goals_minutes'] != '[]':
                try:
                    minutes = ast.literal_eval(game['goals_minutes'])
                    if not isinstance(minutes, list):
                        continue
                    
                    player_name = game['player_name']
                    
                    # Zähle Tore in verschiedenen Zeiträumen
                    for minute in minutes:
                        if isinstance(minute, (int, float)):
                            minute = int(minute)
                            
                            # Verschiedene "späte" Kategorien
                            categories = []
                            if minute >= 80:  # Letzte 10+ Minuten
                                categories.append('last_10_min')
                            if minute >= 85:  # Letzte 5+ Minuten  
                                categories.append('last_5_min')
                            if minute >= 90:  # Nachspielzeit
                                categories.append('injury_time')
                            
                            for category in categories:
                                key = f"{player_name}_{category}"
                                late_game_goals[key] = late_game_goals.get(key, 0) + 1
                                
                except (ValueError, SyntaxError, TypeError):
                    continue
        
        # Organisiere Ergebnisse - TYPE-SAFE
        result = {
            'last_10_min': {},
            'last_5_min': {},
            'injury_time': {}
        }
        
        for key, count in late_game_goals.items():
            parts = key.rsplit('_', 2)  # Split am letzten Unterstrich
            if len(parts) >= 2:
                player_name = '_'.join(parts[:-2]) if len(parts) > 2 else parts[0]
                category = '_'.join(parts[-2:])
                
                if category in result:
                    result[category][player_name] = count
        
        # Sortiere nach Häufigkeit - TYPE-SAFE
        for category in result:
            valid_items = []
            for player, count in result[category].items():
                if isinstance(player, str) and isinstance(count, (int, float)) and not pd.isna(count):
                    valid_items.append((player, int(count)))
            result[category] = sorted(valid_items, key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "late_game_analysis": result,
            "total_last_10_min_goals": sum(count for _, count in result['last_10_min']),
            "total_injury_time_goals": sum(count for _, count in result['injury_time'])
        }

    def get_goals_by_timeframe_with_teammates(self, player_name):
        """Analyse wann Spieler mit welchen Mitspielern trifft"""
        import ast
        
        player_data = self.df[self.df['player_name'] == player_name]
        
        if len(player_data) == 0:
            return None
        
        # Sammle alle Tore mit Zeitangaben
        teammate_timeframe_goals = {
            'first_half': {},  # 1-45 Min
            'second_half': {},  # 46-90 Min
            'extra_time': {}   # 90+ Min
        }
        
        for _, game in player_data.iterrows():
            if game['goals'] > 0 and pd.notna(game['goals_minutes']) and game['goals_minutes'] != '[]':
                try:
                    # Parse Tor-Minuten
                    minutes = ast.literal_eval(game['goals_minutes'])
                    if not isinstance(minutes, list):
                        continue
                        
                    # Finde Mitspieler in diesem Spiel
                    event_id = game['event_id']
                    team_id = game['team_id_wp']
                    teammates = self.df[
                        (self.df['event_id'] == event_id) & 
                        (self.df['team_id_wp'] == team_id) & 
                        (self.df['player_name'] != player_name)
                    ]['player_name'].unique()
                    
                    # Kategorisiere Tore nach Halbzeiten
                    for minute in minutes:
                        if isinstance(minute, (int, float)):
                            minute = int(minute)
                            
                            if 1 <= minute <= 45:
                                timeframe = 'first_half'
                            elif 46 <= minute <= 90:
                                timeframe = 'second_half'
                            else:
                                timeframe = 'extra_time'
                            
                            # Zähle für jeden Mitspieler
                            for teammate in teammates:
                                if teammate not in teammate_timeframe_goals[timeframe]:
                                    teammate_timeframe_goals[timeframe][teammate] = 0
                                teammate_timeframe_goals[timeframe][teammate] += 1
                                
                except (ValueError, SyntaxError, TypeError):
                    continue
        
        # Sortiere nach Häufigkeit - TYPE-SAFE
        result = {}
        for timeframe, teammate_goals in teammate_timeframe_goals.items():
            valid_teammates = []
            for teammate, goals in teammate_goals.items():
                if isinstance(teammate, str) and isinstance(goals, (int, float)) and not pd.isna(goals):
                    valid_teammates.append((teammate, int(goals)))
            result[timeframe] = sorted(valid_teammates, key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "player": player_name,
            "timeframe_analysis": result,
            "total_first_half_goals": sum(teammate_timeframe_goals['first_half'].values()),
            "total_second_half_goals": sum(teammate_timeframe_goals['second_half'].values()),
            "total_extra_time_goals": sum(teammate_timeframe_goals['extra_time'].values())
        }

    def get_trio_stats(self, min_games=5):
        """Findet die besten Dreier-Kombinationen"""
        from collections import defaultdict
        
        trio_stats = defaultdict(lambda: {'games': 0, 'wins': 0, 'losses': 0, 'draws': 0})
        
        # Gruppiere nach Event und Team
        for (event_id, team_id), team_group in self.df.groupby(['event_id', 'team_id_wp']):
            players = sorted(team_group['player_name'].unique())
            
            # Nur Teams mit mindestens 3 Spielern
            if len(players) >= 3:
                result = team_group['result'].iloc[0]  # Alle Spieler haben gleiches result
                
                # Alle 3er-Kombinationen in diesem Team
                from itertools import combinations
                for trio in combinations(players, 3):
                    trio_key = tuple(sorted(trio))
                    
                    trio_stats[trio_key]['games'] += 1
                    if result == 'W':
                        trio_stats[trio_key]['wins'] += 1
                    elif result == 'L':
                        trio_stats[trio_key]['losses'] += 1
                    elif result == 'D':
                        trio_stats[trio_key]['draws'] += 1
        
        # Berechne Siegraten für Trios mit genügend Spielen
        processed_trios = {}
        for trio, stats in trio_stats.items():
            if stats['games'] >= min_games:
                win_rate = round(stats['wins'] / stats['games'] * 100, 1)
                processed_trios[trio] = {
                    **stats,
                    'win_rate': win_rate,
                    'players': list(trio)
                }
        
        # Sortiere nach Siegrate - TYPE-SAFE
        valid_trios = []
        for trio, data in processed_trios.items():
            if isinstance(data.get('win_rate'), (int, float)) and isinstance(data.get('games'), (int, float)):
                valid_trios.append((trio, data))
        
        best_trios = sorted(valid_trios, key=lambda x: (x[1]['win_rate'], x[1]['games']), reverse=True)
        
        return {
            "total_trios_found": len(processed_trios),
            "min_games_threshold": min_games,
            "best_trios": best_trios[:15],  # Top 15 Trios
            "top_by_games": sorted(valid_trios, key=lambda x: x[1]['games'], reverse=True)[:10]
        }

    def get_quartet_stats(self, min_games=3):
        """Findet die besten Vierer-Kombinationen"""
        from collections import defaultdict
        
        quartet_stats = defaultdict(lambda: {'games': 0, 'wins': 0, 'losses': 0, 'draws': 0})
        
        # Gruppiere nach Event und Team
        for (event_id, team_id), team_group in self.df.groupby(['event_id', 'team_id_wp']):
            players = sorted(team_group['player_name'].unique())
            
            # Nur Teams mit mindestens 4 Spielern
            if len(players) >= 4:
                result = team_group['result'].iloc[0]
                
                # Alle 4er-Kombinationen in diesem Team
                from itertools import combinations
                for quartet in combinations(players, 4):
                    quartet_key = tuple(sorted(quartet))
                    
                    quartet_stats[quartet_key]['games'] += 1
                    if result == 'W':
                        quartet_stats[quartet_key]['wins'] += 1
                    elif result == 'L':
                        quartet_stats[quartet_key]['losses'] += 1
                    elif result == 'D':
                        quartet_stats[quartet_key]['draws'] += 1
        
        # Berechne Siegraten
        processed_quartets = {}
        for quartet, stats in quartet_stats.items():
            if stats['games'] >= min_games:
                win_rate = round(stats['wins'] / stats['games'] * 100, 1)
                processed_quartets[quartet] = {
                    **stats,
                    'win_rate': win_rate,
                    'players': list(quartet)
                }
        
        # TYPE-SAFE sorting
        valid_quartets = []
        for quartet, data in processed_quartets.items():
            if isinstance(data.get('win_rate'), (int, float)) and isinstance(data.get('games'), (int, float)):
                valid_quartets.append((quartet, data))
        
        best_quartets = sorted(valid_quartets, key=lambda x: (x[1]['win_rate'], x[1]['games']), reverse=True)
        
        return {
            "total_quartets_found": len(processed_quartets),
            "min_games_threshold": min_games,
            "best_quartets": best_quartets[:10]
        }

    def get_goals_with_teammates(self, player_name):
        """Analyse mit welchen Mitspielern ein Spieler die meisten Tore schießt"""
        player_data = self.df[self.df['player_name'] == player_name]
        
        if len(player_data) == 0:
            return None
        
        teammate_goals = {}
        
        for _, game in player_data.iterrows():
            if game['goals'] > 0:
                # Finde Mitspieler in diesem Spiel
                event_id = game['event_id']
                team_id = game['team_id_wp']
                teammates = self.df[
                    (self.df['event_id'] == event_id) & 
                    (self.df['team_id_wp'] == team_id) & 
                    (self.df['player_name'] != player_name)
                ]['player_name'].unique()
                
                goals_count = int(game['goals']) if pd.notna(game['goals']) else 0
                
                # Zähle Tore mit jedem Mitspieler
                for teammate in teammates:
                    if teammate not in teammate_goals:
                        teammate_goals[teammate] = 0
                    teammate_goals[teammate] += goals_count
        
        # Sortiere sicher - TYPE-SAFE
        sorted_goals = []
        for teammate, goals in teammate_goals.items():
            if isinstance(teammate, str) and isinstance(goals, (int, float)) and not pd.isna(goals):
                sorted_goals.append((teammate, int(goals)))
        
        sorted_goals.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "player": player_name,
            "goals_with_teammates": sorted_goals[:10],
            "total_goals": int(player_data['goals'].sum()),
            "teammates_analyzed": len(sorted_goals)
        }

    def get_assists_with_teammates(self, player_name):
        """Analyse mit welchen Mitspielern ein Spieler die meisten Assists hat"""
        player_data = self.df[self.df['player_name'] == player_name]
        
        if len(player_data) == 0:
            return None
        
        teammate_assists = {}
        
        for _, game in player_data.iterrows():
            if game['assists'] > 0:
                # Finde Mitspieler in diesem Spiel
                event_id = game['event_id']
                team_id = game['team_id_wp']
                teammates = self.df[
                    (self.df['event_id'] == event_id) & 
                    (self.df['team_id_wp'] == team_id) & 
                    (self.df['player_name'] != player_name)
                ]['player_name'].unique()
                
                assists_count = int(game['assists']) if pd.notna(game['assists']) else 0
                
                # Verteile Assists auf alle Mitspieler
                for teammate in teammates:
                    if teammate not in teammate_assists:
                        teammate_assists[teammate] = 0
                    teammate_assists[teammate] += assists_count
        
        # Sortiere sicher - TYPE-SAFE
        sorted_assists = []
        for teammate, assists in teammate_assists.items():
            if isinstance(teammate, str) and isinstance(assists, (int, float)) and not pd.isna(assists):
                sorted_assists.append((teammate, int(assists)))
        
        sorted_assists.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "player": player_name,
            "assists_with_teammates": sorted_assists[:10],
            "total_assists": int(player_data['assists'].sum()),
            "teammates_analyzed": len(sorted_assists)
        }

    def get_all_teammate_winrates(self, player_name):
        """Siegraten mit allen Mitspielern berechnen"""
        player_data = self.df[self.df['player_name'] == player_name]
        
        if len(player_data) == 0:
            return None
        
        # Alle Mitspieler-Kombinationen sammeln
        teammate_stats = {}
        player_events = set(zip(player_data['event_id'], player_data['team_id_wp']))
        
        for event_id, team_id in player_events:
            # Spieler-Performance in diesem Spiel
            player_game = player_data[player_data['event_id'] == event_id].iloc[0]
            game_result = player_game['result']
            
            # Alle Mitspieler in diesem Spiel
            same_team = self.df[(self.df['event_id'] == event_id) & (self.df['team_id_wp'] == team_id)]
            
            for _, teammate in same_team.iterrows():
                teammate_name = teammate['player_name']
                if teammate_name != player_name:
                    
                    if teammate_name not in teammate_stats:
                        teammate_stats[teammate_name] = {
                            'games': 0, 'wins': 0, 'losses': 0, 'draws': 0
                        }
                    
                    teammate_stats[teammate_name]['games'] += 1
                    
                    if game_result == 'W':
                        teammate_stats[teammate_name]['wins'] += 1
                    elif game_result == 'L':
                        teammate_stats[teammate_name]['losses'] += 1
                    elif game_result == 'D':
                        teammate_stats[teammate_name]['draws'] += 1
        
        # Berechne Siegraten (min. 5 Spiele für Aussagekraft)
        processed_stats = {}
        for teammate, stats in teammate_stats.items():
            if stats['games'] >= 5:
                win_rate = round(stats['wins'] / stats['games'] * 100, 1)
                processed_stats[teammate] = {
                    **stats,
                    'win_rate': win_rate,
                    'loss_rate': round(stats['losses'] / stats['games'] * 100, 1)
                }
        
        # Sortierungen - TYPE-SAFE
        valid_stats = [(k, v) for k, v in processed_stats.items() if isinstance(v.get('win_rate'), (int, float))]
        by_win_rate = sorted(valid_stats, key=lambda x: x[1]['win_rate'], reverse=True)
        by_games = sorted(valid_stats, key=lambda x: x[1]['games'], reverse=True)
        
        return {
            "player": player_name,
            "total_teammates_analyzed": len(processed_stats),
            "best_win_rate_with": by_win_rate[:10],  # Top 10 nach Siegrate
            "most_games_with": by_games[:10],  # Top 10 nach Spielanzahl
            "detailed_stats": processed_stats
        }

    def get_player_complete_stats(self, player_name):
        """Vollständige Statistiken für einen Spieler"""
        player_data = self.df[self.df['player_name'] == player_name]
        
        if len(player_data) == 0:
            return None
        
        # Basis-Stats
        stats = {
            "name": player_name,
            "games": len(player_data),
            "goals": int(player_data['goals'].sum()),
            "assists": int(player_data['assists'].sum()),
            "wins": int((player_data['result'] == 'W').sum()),
            "losses": int((player_data['result'] == 'L').sum()),
            "draws": int((player_data['result'] == 'D').sum())
        }
        
        if stats["games"] > 0:
            stats["win_rate"] = round(stats["wins"] / stats["games"] * 100, 1)
            stats["goals_per_game"] = round(stats["goals"] / stats["games"], 2)
            stats["assists_per_game"] = round(stats["assists"] / stats["games"], 2)
        
        # Jahresstatistiken
        player_data_copy = player_data.copy()
        player_data_copy['date'] = pd.to_datetime(player_data_copy['date'], errors='coerce')
        player_data_copy['year'] = player_data_copy['date'].dt.year
        
        yearly_stats = {}
        for year, year_data in player_data_copy.groupby('year'):
            if pd.notna(year):
                yearly_stats[int(year)] = {
                    "goals": int(year_data['goals'].sum()),
                    "assists": int(year_data['assists'].sum()),
                    "games": len(year_data),
                    "wins": int((year_data['result'] == 'W').sum())
                }
        stats["yearly_stats"] = yearly_stats
        
        # Top Mitspieler
        player_events = set(zip(player_data['event_id'], player_data['team_id_wp']))
        teammates = {}
        
        for event_id, team_id in player_events:
            same_team = self.df[(self.df['event_id'] == event_id) & (self.df['team_id_wp'] == team_id)]
            for _, teammate in same_team.iterrows():
                teammate_name = teammate['player_name']
                if teammate_name != player_name:
                    teammates[teammate_name] = teammates.get(teammate_name, 0) + 1
        
        # Top 10 häufigste Mitspieler - TYPE-SAFE
        valid_teammates = [(k, v) for k, v in teammates.items() if isinstance(k, str) and isinstance(v, (int, float))]
        top_teammates = sorted(valid_teammates, key=lambda x: x[1], reverse=True)[:10]
        stats["frequent_teammates"] = top_teammates
        
        return stats
    
    def get_combination_stats(self, player1, player2):
        """Mitspieler-Statistiken"""
        p1_data = self.df[self.df['player_name'] == player1]
        p2_data = self.df[self.df['player_name'] == player2]
        
        if len(p1_data) == 0 or len(p2_data) == 0:
            return None
        
        # Gemeinsame Spiele finden
        p1_events = set(zip(p1_data['event_id'], p1_data['team_id_wp']))
        p2_events = set(zip(p2_data['event_id'], p2_data['team_id_wp']))
        common_events = p1_events & p2_events
        
        if not common_events:
            return {"note": f"{player1} und {player2} haben nie zusammen gespielt"}
        
        # Stats für gemeinsame Spiele
        common_event_ids = [e[0] for e in common_events]
        p1_together = p1_data[p1_data['event_id'].isin(common_event_ids)]
        p1_without = p1_data[~p1_data['event_id'].isin(common_event_ids)]
        
        stats = {
            "player1": player1,
            "player2": player2,
            "games_together": len(common_events),
            "wins_together": int((p1_together['result'] == 'W').sum()),
            "losses_together": int((p1_together['result'] == 'L').sum()),
            "draws_together": int((p1_together['result'] == 'D').sum()),
            "p1_goals_together": int(p1_together['goals'].sum()),
            "p1_goals_avg_together": round(p1_together['goals'].mean(), 2),
            "p1_winrate_together": round((p1_together['result'] == 'W').sum() / len(p1_together) * 100, 1)
        }
        
        # Vergleich ohne Partner
        if len(p1_without) > 0:
            stats["p1_winrate_without"] = round((p1_without['result'] == 'W').sum() / len(p1_without) * 100, 1)
            stats["p1_goals_avg_without"] = round(p1_without['goals'].mean(), 2)
        
        return stats
    
    def get_head_to_head_stats(self, player1, player2):
        """Gegner-Statistiken"""
        p1_data = self.df[self.df['player_name'] == player1]
        p2_data = self.df[self.df['player_name'] == player2]
        
        if len(p1_data) == 0 or len(p2_data) == 0:
            return None
        
        # Gemeinsame Events finden
        p1_events = set(p1_data['event_id'])
        p2_events = set(p2_data['event_id'])
        common_events = p1_events & p2_events
        
        if not common_events:
            return {"note": f"{player1} und {player2} haben nie gegeneinander gespielt"}
        
        # Prüfe welche Spiele als Gegner (verschiedene Teams)
        head_to_head = []
        for event_id in common_events:
            p1_game = p1_data[p1_data['event_id'] == event_id]
            p2_game = p2_data[p2_data['event_id'] == event_id]
            
            if len(p1_game) > 0 and len(p2_game) > 0:
                p1_team = p1_game['team_id_wp'].iloc[0]
                p2_team = p2_game['team_id_wp'].iloc[0]
                
                if p1_team != p2_team:  # Verschiedene Teams = Gegner
                    head_to_head.append({
                        'p1_result': p1_game['result'].iloc[0],
                        'p1_goals': int(p1_game['goals'].iloc[0]),
                        'p2_goals': int(p2_game['goals'].iloc[0])
                    })
        
        if not head_to_head:
            return {"note": f"{player1} und {player2} waren nie direkte Gegner - nur Mitspieler"}
        
        stats = {
            "player1": player1,
            "player2": player2,
            "head_to_head_games": len(head_to_head),
            "p1_wins": len([g for g in head_to_head if g['p1_result'] == 'W']),
            "p1_losses": len([g for g in head_to_head if g['p1_result'] == 'L']),
            "p1_draws": len([g for g in head_to_head if g['p1_result'] == 'D']),
            "p1_goals_total": sum(g['p1_goals'] for g in head_to_head),
            "p2_goals_total": sum(g['p2_goals'] for g in head_to_head)
        }
        
        if stats["head_to_head_games"] > 0:
            stats["p1_win_rate"] = round(stats["p1_wins"] / stats["head_to_head_games"] * 100, 1)
        
        return stats
    
    def analyze_question(self, question):
        """Intelligente Fragen-Analyse"""
        context = {}
        question_lower = question.lower()
        
        # 1. Finde alle Spieler in der Frage
        found_players = self.find_players_in_question(question)
        
        # 2. Lade Basis-Stats für gefundene Spieler
        for player in found_players:
            safe_name = player.replace(' ', '_').replace('.', '').lower()
            context[f"{safe_name}_stats"] = self.get_player_complete_stats(player)
        
        # 3. Erkenne Beziehungstyp und lade passende Daten
        if any(word in question_lower for word in ['gegen', 'vs', 'gegeneinander', 'direkten']):
            # Head-to-Head gefragt
            if len(found_players) >= 2:
                p1, p2 = found_players[0], found_players[1]
                context["head_to_head"] = self.get_head_to_head_stats(p1, p2)
                
        elif any(word in question_lower for word in ['mit', 'zusammen', 'kombination', 'duo', 'erfolgreich']):
            # Mitspieler-Combo gefragt - auch bei allgemeinen Fragen
            if len(found_players) >= 2:
                p1, p2 = found_players[0], found_players[1]
                context["combination"] = self.get_combination_stats(p1, p2)
            elif any(word in question_lower for word in ['erfolgreichste', 'beste', 'duo']):
                # Allgemeine "beste Kombination" Frage - lade Trio-Daten
                context["trio_analysis"] = self.get_trio_stats()
                # Zusätzlich: finde häufigste Spieler-Paare
                top_players = self.df['player_name'].value_counts().head(10).index.tolist()
                context["frequent_players"] = top_players
        
        # 4. Erkenne spezielle Fragen
        if any(word in question_lower for word in ['tore', 'trifft', 'schießt', 'goals']):
            # Tor-Analysen mit Mitspielern
            for player in found_players:
                safe_name = player.replace(' ', '_').replace('.', '').lower()
                context[f"{safe_name}_goals_teammates"] = self.get_goals_with_teammates(player)
                
        elif any(word in question_lower for word in ['assists', 'vorlagen', 'vorlage', 'teamkameraden']):
            # Assists-Analysen mit Mitspielern
            for player in found_players:
                safe_name = player.replace(' ', '_').replace('.', '').lower()
                context[f"{safe_name}_assists_teammates"] = self.get_assists_with_teammates(player)
                
        elif any(word in question_lower for word in ['minuten', 'zeit', 'interval', 'phase', 'alle analysen', 'vollständig']):
            # Umfassende Zeitintervall-Analysen
            context["comprehensive_time_analysis"] = self.get_comprehensive_time_analysis()
            
        elif any(word in question_lower for word in ['letzten', 'späten', 'nachspielzeit', 'schluss', 'injury time']):
            # Spezifische späte Tor-Analysen
            context["late_game_scorers"] = self.get_late_game_scorers()
            context["comprehensive_time_analysis"] = self.get_comprehensive_time_analysis()  # Auch vollständig für Kontext
            
        elif any(word in question_lower for word in ['erste halbzeit', 'zweite halbzeit', 'halbzeit', 'minute', 'wann trifft']):
            # Zeitbasierte Tor-Analysen für spezifische Spieler
            for player in found_players:
                safe_name = player.replace(' ', '_').replace('.', '').lower()
                context[f"{safe_name}_timeframe_goals"] = self.get_goals_by_timeframe_with_teammates(player)
                
        elif any(word in question_lower for word in ['trio', 'dreier', '3er', 'drei spieler']):
            # Trio-Analysen
            context["trio_analysis"] = self.get_trio_stats()
            
        elif any(word in question_lower for word in ['quartett', 'vierer', '4er', 'vier spieler']):
            # Quartett-Analysen  
            context["quartet_analysis"] = self.get_quartet_stats()
            
        elif any(word in question_lower for word in ['kombination', 'team', 'gruppe', 'formation']):
            # Allgemeine Team-Kombinationen (Trio + Quartett)
            context["trio_analysis"] = self.get_trio_stats()
            if any(word in question_lower for word in ['groß', 'vier', 'mehr']):
                context["quartet_analysis"] = self.get_quartet_stats()
                
        elif any(word in question_lower for word in ['prozentual', 'siegrate', 'gewinnrate', 'häufigsten', 'besten mitspieler']):
            # Detaillierte Mitspieler-Siegraten
            for player in found_players:
                safe_name = player.replace(' ', '_').replace('.', '').lower()
                context[f"{safe_name}_teammate_winrates"] = self.get_all_teammate_winrates(player)
                
        elif any(word in question_lower for word in ['beste', 'top', 'torschütz']):
            # Top Listen
            context["top_scorers"] = self.df.groupby('player_name')['goals'].sum().sort_values(ascending=False).head(10).to_dict()
            
        if any(word in question_lower for word in ['jahr', '202']):
            # Jahr extrahieren
            years = re.findall(r'20\d{2}', question)
            if years:
                year = int(years[0])
                year_data = self.df.copy()
                year_data['date'] = pd.to_datetime(year_data['date'], errors='coerce')
                year_data = year_data[year_data['date'].dt.year == year]
                if len(year_data) > 0:
                    context["year_data"] = {
                        "year": year,
                        "top_scorers": year_data.groupby('player_name')['goals'].sum().sort_values(ascending=False).head(5).to_dict()
                    }
        
        # Fallback: Bei leeren found_players aber relevanten Fragen trotzdem Daten laden
        if not found_players and any(word in question_lower for word in 
                                   ['beste', 'erfolgreichste', 'top', 'häufigsten', 'alle', 'wer']):
            # Allgemeine Statistikfragen - lade umfassende Daten
            context["trio_analysis"] = self.get_trio_stats()
            context["comprehensive_time_analysis"] = self.get_comprehensive_time_analysis()
            context["top_scorers"] = self.df.groupby('player_name')['goals'].sum().sort_values(ascending=False).head(15).to_dict()
            context["top_assisters"] = self.df.groupby('player_name')['assists'].sum().sort_values(ascending=False).head(10).to_dict()
            
            # Häufigste Spieler für Kontext
            top_players = self.df['player_name'].value_counts().head(15).index.tolist()
            context["most_active_players"] = [(player, int(count)) for player, count in 
                                            self.df['player_name'].value_counts().head(15).items()]
        
        return context
    
    def ask_claude(self, question, context):
        """Frage an Claude mit intelligentem Kontext"""
        prompt = f"""Du bist ein Fußball-Datenanalyst. Beantworte die Frage basierend auf den bereitgestellten Daten.

VERFÜGBARE DATEN:
{json.dumps(context, indent=2, ensure_ascii=False, default=str)}

DATENSCHEMA:
- CSV mit 2594 Spieler-Einträgen aus {len(self.all_players)} verschiedenen Spielern
- Spalten: player_name, goals, assists, result (W/L/D), event_id, team_id_wp (Leibchen 1-3), date
- yearly_stats = Jahresaufschlüsselung pro Spieler
- combination = Mitspieler-Stats (gleiches Team)
- head_to_head = Gegner-Stats (verschiedene Teams)

FRAGE: {question}

Gib eine präzise, datenbasierte Antwort auf Deutsch:"""

        try:
            response = self.client.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=400,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return f"❌ Claude Fehler: {e}"
    
    def chat(self):
        """Interaktiver Chat"""
        print(f"\n🤖 Smart Football Analyzer bereit!")
        print(f"📊 {len(self.all_players)} Spieler erkannt: {', '.join(self.all_players[:5])}...")
        print("\nStellen Sie beliebige Fragen über jeden Spieler!\n")
        
        while True:
            try:
                question = input("❓ Frage: ").strip()
                
                if question.lower() in ['quit', 'exit', 'bye']:
                    print("Auf Wiedersehen!")
                    break
                    
                if not question:
                    continue
                
                print("🔍 Analysiere Frage und suche relevante Daten...")
                
                # Intelligente Analyse
                context = self.analyze_question(question)
                
                # Zeige gefundene Spieler (Debug)
                found_players = self.find_players_in_question(question)
                if found_players:
                    print(f"   Gefundene Spieler: {', '.join(found_players)}")
                
                # Claude fragen
                answer = self.ask_claude(question, context)
                print(f"💡 {answer}\n")
                
            except KeyboardInterrupt:
                print("\nAuf Wiedersehen!")
                break
            except Exception as e:
                print(f"❌ Fehler: {e}")

def main():
    analyzer = SmartFootballAnalyzer()
    
    if not analyzer.load_data():
        return
    if not analyzer.setup_claude():
        return
        
    analyzer.chat()

if __name__ == "__main__":
    main()
