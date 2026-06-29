# -*- coding: utf-8 -*-
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import os

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ----- 1. 전체 댓글 수집 -----
print('[+] 댓글 수집 중...')
total_comments = []
for post_id in tqdm(range(1, 101)):
    url = f'http://board.nyan101.com/comments/{post_id}'
    response = requests.get(url)
    total_comments.extend(response.json())
print(f'[+] 총 {len(total_comments)}개 댓글 수집 완료')
print()

# ----- 2. 사용자별로 분류 -----
user_comments = {}
for comment in total_comments:
    author_id = comment['author_id']
    if author_id not in user_comments:
        user_comments[author_id] = []
    user_comments[author_id].append(comment)

# ----- 3. 비정상 패턴 탐지 -----
print('[+] 비정상 패턴 탐지 중...')
print('=' * 60)

abnormal_users = []

for author_id, comments in user_comments.items():
    author_name = comments[0]['author_name']
    reasons = []

    timestamps = sorted([
        datetime.strptime(c['created_at'], '%Y-%m-%d %H:%M')
        for c in comments
    ])

    diffs = [
        int((timestamps[i+1] - timestamps[i]).total_seconds() // 60)
        for i in range(len(timestamps) - 1)
    ]

    # 기준 1: 최대 연속 휴식 시간 180분 미만
    if diffs:
        max_rest = max(diffs)
        if max_rest < 180:
            reasons.append(f'[기준1] 24시간 활성화: 최대 휴식 {max_rest}분 (기준: 180분 미만)')

    # 기준 2: 1분 도배
    minute_counts = defaultdict(int)
    for dt in timestamps:
        minute_counts[dt.strftime('%Y-%m-%d %H:%M')] += 1
    max_per_minute = max(minute_counts.values())
    if max_per_minute >= 5:
        worst_minute = max(minute_counts, key=minute_counts.get)
        reasons.append(f'[기준2] 1분 도배: {worst_minute}에 {max_per_minute}개 작성')

    # 기준 3: 고정 간격 (표준편차)
    valid_diffs = [d for d in diffs if d > 0]
    if len(valid_diffs) >= 5:
        std = np.std(valid_diffs)
        if std < 1.5:
            reasons.append(f'[기준3] 고정 간격: 댓글 간격 표준편차 {std:.2f}분 (기준: 1.5 미만)')

    # 기준 4: 댓글 내용 중복
    content_counts = defaultdict(int)
    for c in comments:
        content_counts[c['content']] += 1
    max_duplicate = max(content_counts.values())
    if max_duplicate >= 3:
        reasons.append(f'[기준4] 내용 중복: 동일 댓글 {max_duplicate}회 반복')

    # 기준 5: 5분 배수 집중
    round_minutes = sum(1 for dt in timestamps if dt.minute % 5 == 0)
    round_ratio = round_minutes / len(timestamps)
    if round_ratio >= 0.8:
        reasons.append(f'[기준5] 정각 집중: 5분 배수 비율 {round_ratio:.0%}')

    if reasons:
        abnormal_users.append((author_id, author_name, len(comments), reasons, timestamps, comments))

# ----- 4. 결과 출력 -----
print(f'[+] 전체 {len(user_comments)}명 중 비정상 유저 {len(abnormal_users)}명 탐지')
print('=' * 60)
for author_id, author_name, count, reasons, _, __ in abnormal_users:
    print(f'\n[!] {author_name} (ID: {author_id}) - 총 {count}개 댓글')
    for reason in reasons:
        print(f'    {reason}')
print()

# ----- 5. 시각화 -----
os.makedirs('q2_output', exist_ok=True)
weekdays = ['월', '화', '수', '목', '금', '토', '일']
hours = [f'{i:02d}' for i in range(24)]

for author_id, author_name, count, reasons, timestamps, comments in abnormal_users:

    reason_labels = '\n'.join(reasons)

    # 기준2(1분 도배)에 걸린 경우 → 타임라인 바 차트
    is_burst = any('[기준2]' in r for r in reasons)

    if is_burst:
        # 분별 댓글 수 집계
        minute_counts = defaultdict(int)
        for dt in timestamps:
            minute_counts[dt.strftime('%Y-%m-%d %H:%M')] += 1

        sorted_minutes = sorted(minute_counts.keys())
        counts = [minute_counts[m] for m in sorted_minutes]
        colors = ['#d32f2f' if c >= 5 else '#388e3c' for c in counts]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(range(len(sorted_minutes)), counts, color=colors)

        # X축: 날짜 레이블 (많으면 일부만 표시)
        step = max(1, len(sorted_minutes) // 20)
        ax.set_xticks(range(0, len(sorted_minutes), step))
        ax.set_xticklabels(
            [sorted_minutes[i] for i in range(0, len(sorted_minutes), step)],
            rotation=45, ha='right', fontsize=7
        )
        ax.set_ylabel('댓글 수')
        ax.set_xlabel('작성 시간 (분 단위)')

        # 빨간 칸에 숫자 표시
        for i, (c, bar) in enumerate(zip(counts, bars)):
            if c >= 5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        str(c), ha='center', va='bottom',
                        color='#d32f2f', fontweight='bold', fontsize=9)

        ax.set_title(
            f'[비정상] {author_name} (ID: {author_id}) - 총 {count}개 댓글\n'
            f'{reason_labels}\n'
            f'※ 빨간 막대 = 1분 내 5개 이상 작성 구간',
            fontsize=9, loc='left'
        )
        plt.tight_layout()
        plt.savefig(f'q2_output/{author_id}_{author_name}.png', dpi=150, bbox_inches='tight')
        plt.close()

    else:
        # 기준2 외 → 히트맵
        matrix = np.zeros((24, 7), dtype=int)
        for dt in timestamps:
            matrix[dt.hour, dt.weekday()] += 1

        diffs = [
            int((timestamps[i+1] - timestamps[i]).total_seconds() // 60)
            for i in range(len(timestamps) - 1)
        ]
        valid_diffs = [d for d in diffs if d > 0]
        avg_interval = f'{np.mean(valid_diffs):.1f}분' if valid_diffs else '-'
        std_interval = f'{np.std(valid_diffs):.2f}분' if valid_diffs else '-'

        fig, ax = plt.subplots(figsize=(6, 10))
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')
        plt.colorbar(im, ax=ax, label='댓글 수')
        ax.set_xticks(range(7))
        ax.set_xticklabels(weekdays)
        ax.set_yticks(range(24))
        ax.set_yticklabels(hours)

        for h in range(24):
            for w in range(7):
                if matrix[h, w] > 0:
                    ax.text(w, h, str(matrix[h, w]),
                            ha='center', va='center', color='white', fontsize=8)

        ax.set_title(
            f'[비정상] {author_name} (ID: {author_id}) - 총 {count}개 댓글\n'
            f'{reason_labels}\n'
            f'평균 간격: {avg_interval} / 표준편차: {std_interval}',
            fontsize=9, loc='left'
        )
        plt.tight_layout()
        plt.savefig(f'q2_output/{author_id}_{author_name}.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f'  [+] {author_name} 저장 완료')

print()
print('=' * 60)
print(f'[+] 완료! q2_output 폴더에 {len(abnormal_users)}개 파일 저장됨')