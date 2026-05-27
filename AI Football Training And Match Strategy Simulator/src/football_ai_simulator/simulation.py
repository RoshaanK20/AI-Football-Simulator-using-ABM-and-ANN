from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from .ann import MatchPredictor, Prediction
from .data import MatchSituation


WIDTH = 1100
HEIGHT = 680
PITCH = pygame.Rect(55, 70, 990, 560)
GOAL_DEPTH = 26
GOAL_MOUTH_HALF_HEIGHT = 58
DEFAULT_MATCH_SECONDS = 120.0
BALL_SPEED = 660.0
DECISION_INTERVAL = 0.85
DEFENDER_PRESS_RADIUS = 155
PASS_INTERCEPT_MIN = 0.35
PASS_INTERCEPT_MAX = 0.94

FIELD_GREEN = (35, 122, 72)
FIELD_DARK = (30, 104, 65)
LINE = (238, 244, 235)
TEAM_A = (214, 51, 74)
TEAM_B = (46, 96, 215)
BALL = (245, 245, 238)
INK = (18, 26, 32)
PANEL = (245, 247, 243)
ACCENT = (238, 194, 74)


@dataclass(frozen=True)
class MatchSetup:
    red_target_score: int = 2
    blue_target_score: int = 1
    red_passes_per_goal: int = 4
    blue_passes_per_goal: int = 4

    def target_score(self, team: str) -> int:
        return self.red_target_score if team == "A" else self.blue_target_score

    def passes_per_goal(self, team: str) -> int:
        return self.red_passes_per_goal if team == "A" else self.blue_passes_per_goal


TEAM_NAMES = {"A": "Red", "B": "Blue"}


@dataclass
class PlayerAgent:
    number: int
    team: str
    role: str
    position: pygame.Vector2
    home: pygame.Vector2
    passing_accuracy: float
    stamina: float = 100.0
    has_ball: bool = False

    @property
    def color(self) -> tuple[int, int, int]:
        return TEAM_A if self.team == "A" else TEAM_B

    @property
    def attack_direction(self) -> int:
        return 1 if self.team == "A" else -1

    def distance_to(self, other: pygame.Vector2) -> float:
        return self.position.distance_to(other)

    def update(self, sim: "FootballSimulation", dt: float) -> None:
        target = pygame.Vector2(self.home)

        if self.has_ball:
            target.x += self.attack_direction * 92
        elif sim.ball_owner and sim.ball_owner.team == self.team:
            target.x += self.attack_direction * 44
            target.y += math.sin(sim.elapsed * 1.3 + self.number) * 18
        else:
            target = self.position.lerp(sim.ball_position, 0.018)
            target.x -= self.attack_direction * 26

        if self.stamina < 24:
            target = self.position.lerp(self.home, 0.05)

        target.x = max(PITCH.left + 20, min(PITCH.right - 20, target.x))
        target.y = max(PITCH.top + 20, min(PITCH.bottom - 20, target.y))

        delta = target - self.position
        if delta.length_squared() > 3:
            speed = 92 + self.stamina * 0.55
            step = min(delta.length(), speed * dt)
            self.position += delta.normalize() * step
            self.stamina = max(0.0, self.stamina - dt * (0.9 if self.has_ball else 0.42))
        else:
            self.stamina = min(100.0, self.stamina + dt * 1.4)


class FootballSimulation:
    def __init__(
        self,
        predictor: MatchPredictor,
        seed: int = 7,
        match_seconds: float = DEFAULT_MATCH_SECONDS,
        match_setup: MatchSetup | None = None,
    ) -> None:
        random.seed(seed)
        self.predictor = predictor
        self.match_seconds = max(15.0, match_seconds)
        self.match_setup = match_setup or MatchSetup()
        self.elapsed = 0.0
        self.score_a = 0
        self.score_b = 0
        self.completed_passes = {"A": 0, "B": 0}
        self.players = self._create_players()
        self.ball_position = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
        self.ball_target: pygame.Vector2 | None = None
        self.pending_owner: PlayerAgent | None = None
        self.pending_pass_team: str | None = None
        self.pending_shot_team: str | None = None
        self.pending_shot_outcome: str | None = None
        self.ball_owner = self.players[0]
        self.ball_owner.has_ball = True
        self.last_prediction = Prediction(0.0, 0.0, 0.0, "Building first read")
        self.last_action = "User match setup ready"
        self.decision_timer = 0.0
        self.flash_timer = 0.0
        self.goal_message = "GOAL"
        self.goal_scoreline = "0 - 0"
        self.paused = False
        self.match_finished = self._targets_reached()
        self.stats = {
            "A": {"passes": 0, "shots": 0, "wide": 0, "turnovers": 0},
            "B": {"passes": 0, "shots": 0, "wide": 0, "turnovers": 0},
        }
        if self.match_finished:
            self.last_action = f"FULL TIME - Red {self.score_a} - {self.score_b} Blue"

    def _create_players(self) -> list[PlayerAgent]:
        setup = [
            ("A", 7, "Winger", (180, 190), 83),
            ("A", 9, "Striker", (255, 340), 78),
            ("A", 10, "Playmaker", (205, 485), 92),
            ("A", 5, "Midfielder", (380, 275), 74),
            ("A", 4, "Defender", (175, 545), 70),
            ("B", 11, "Winger", (920, 190), 80),
            ("B", 8, "Striker", (845, 340), 76),
            ("B", 6, "Playmaker", (895, 485), 88),
            ("B", 3, "Midfielder", (720, 275), 73),
            ("B", 2, "Defender", (925, 545), 71),
        ]
        return [
            PlayerAgent(
                team=team,
                number=number,
                role=role,
                position=pygame.Vector2(pos),
                home=pygame.Vector2(pos),
                passing_accuracy=passing_accuracy,
                stamina=random.uniform(76, 100),
            )
            for team, number, role, pos, passing_accuracy in setup
        ]

    def match_minute(self) -> int:
        return max(1, min(90, int((self.elapsed / self.match_seconds) * 90) + 1))

    def distance_from_goal(self, player: PlayerAgent) -> float:
        goal_x = PITCH.right if player.team == "A" else PITCH.left
        raw = abs(goal_x - player.position.x)
        return max(1.0, min(55.0, raw / PITCH.width * 55.0))

    def nearby_defenders(self, player: PlayerAgent) -> int:
        return sum(
            1
            for other in self.players
            if other.team != player.team and other.distance_to(player.position) <= DEFENDER_PRESS_RADIUS
        )

    def situation_for(self, player: PlayerAgent) -> MatchSituation:
        return MatchSituation(
            stamina=player.stamina,
            distance_goal=self.distance_from_goal(player),
            nearby_defenders=self.nearby_defenders(player),
            passing_accuracy=player.passing_accuracy,
            match_minute=self.match_minute(),
        )

    def update(self, dt: float) -> None:
        if self.paused or self.match_finished:
            return

        self.elapsed += dt
        self.flash_timer = max(0.0, self.flash_timer - dt)

        for player in self.players:
            player.update(self, dt)

        if self.ball_target is not None:
            self._update_free_ball(dt)
        elif self.ball_owner is not None:
            self.ball_position = pygame.Vector2(self.ball_owner.position)
            self.decision_timer -= dt
            if self.decision_timer <= 0:
                self._make_decision(self.ball_owner)
                self.decision_timer = DECISION_INTERVAL

    def _update_free_ball(self, dt: float) -> None:
        assert self.ball_target is not None
        delta = self.ball_target - self.ball_position
        if delta.length() <= 10:
            self.ball_position = pygame.Vector2(self.ball_target)
            self.ball_target = None
            if self.pending_owner is not None:
                if self.pending_shot_outcome == "blocked":
                    shot_team = self.pending_shot_team
                    blocker_team = self.pending_owner.team
                    self.pending_shot_outcome = None
                    self.pending_shot_team = None
                    if shot_team is not None:
                        self.stats[shot_team]["turnovers"] += 1
                        self.last_action = (
                            f"{TEAM_NAMES[blocker_team]} wins ball from early "
                            f"{TEAM_NAMES[shot_team]} shot"
                        )
                if self.pending_pass_team is not None:
                    pass_team = self.pending_pass_team
                    team_name = TEAM_NAMES[pass_team]
                    if self._team_score(pass_team) < self.match_setup.target_score(pass_team):
                        self.completed_passes[pass_team] += 1
                        self.stats[pass_team]["passes"] += 1
                        passes = self.completed_passes[pass_team]
                        needed = self.match_setup.passes_per_goal(pass_team)
                        self.last_action = f"{team_name} random pass {passes}/{needed}"
                    else:
                        self.last_action = f"{team_name} keeps ball; target score reached"
                self.pending_pass_team = None
                self._set_owner(self.pending_owner)
            else:
                self._finish_shot()
            return

        self.ball_position += delta.normalize() * min(delta.length(), BALL_SPEED * dt)

    def _make_decision(self, player: PlayerAgent) -> None:
        situation = self.situation_for(player)
        prediction = self.predictor.predict(situation)
        team = player.team

        if self._ready_to_score(team):
            self.last_prediction = Prediction(
                prediction.pass_success_probability,
                prediction.goal_probability,
                prediction.fatigue_risk,
                "Shoot after user passes",
            )
            self._shoot(player)
            return

        action = random.choices(
            ["pass", "shoot", "carry"],
            weights=self._action_weights(team),
            k=1,
        )[0]

        if action == "pass":
            self.last_prediction = Prediction(
                prediction.pass_success_probability,
                prediction.goal_probability,
                prediction.fatigue_risk,
                "Random pass",
            )
            self._pass(player)
        elif action == "shoot":
            self.last_prediction = Prediction(
                prediction.pass_success_probability,
                prediction.goal_probability,
                prediction.fatigue_risk,
                "Random shot",
            )
            self._shoot(player)
        else:
            self.last_prediction = Prediction(
                prediction.pass_success_probability,
                prediction.goal_probability,
                prediction.fatigue_risk,
                "Random carry",
            )
            team_name = TEAM_NAMES[player.team]
            self.last_action = f"{team_name} carries ball"
            player.home.x += player.attack_direction * 18
            player.home.y += random.choice([-16, 16])
            player.home.x = max(PITCH.left + 40, min(PITCH.right - 40, player.home.x))
            player.home.y = max(PITCH.top + 40, min(PITCH.bottom - 40, player.home.y))

    def _action_weights(self, team: str) -> list[float]:
        if self._team_score(team) >= self.match_setup.target_score(team):
            return [0.58, 0.12, 0.30]

        passes_done = self.completed_passes[team]
        passes_needed = self.match_setup.passes_per_goal(team)
        if passes_done < passes_needed:
            return [0.70, 0.12, 0.18]
        return [0.12, 0.76, 0.12]

    def _team_score(self, team: str) -> int:
        return self.score_a if team == "A" else self.score_b

    def _add_goal(self, team: str) -> None:
        if team == "A":
            self.score_a += 1
        else:
            self.score_b += 1

    def _ready_to_score(self, team: str) -> bool:
        return (
            self._team_score(team) < self.match_setup.target_score(team)
            and self.completed_passes[team] >= self.match_setup.passes_per_goal(team)
        )

    def _targets_reached(self) -> bool:
        return (
            self.score_a >= self.match_setup.red_target_score
            and self.score_b >= self.match_setup.blue_target_score
        )

    def _finish_match_if_ready(self) -> None:
        if self._targets_reached():
            self.match_finished = True
            self.ball_target = None
            self.pending_owner = None
            self.pending_pass_team = None
            self.pending_shot_team = None
            self.pending_shot_outcome = None
            self.last_action = f"FULL TIME - Red {self.score_a} - {self.score_b} Blue"

    def _set_owner(self, player: PlayerAgent) -> None:
        for candidate in self.players:
            candidate.has_ball = False
        self.ball_owner = player
        self.ball_owner.has_ball = True
        self.ball_position = pygame.Vector2(player.position)

    def _pass(self, player: PlayerAgent) -> None:
        teammates = [
            candidate
            for candidate in self.players
            if candidate.team == player.team and candidate is not player
        ]
        forward_teammates = [
            candidate
            for candidate in teammates
            if (candidate.position.x - player.position.x) * player.attack_direction > -20
        ]
        options = forward_teammates or teammates
        target = random.choice(options)
        pass_success_chance = self._pass_success_chance(player)
        self.ball_owner = None
        player.has_ball = False
        if random.random() <= pass_success_chance:
            self.pending_owner = target
            self.pending_pass_team = player.team
            ball_destination = target.position
            action_text = f"{TEAM_NAMES[player.team]} random pass toward #{target.number}"
        else:
            interceptor = self._nearest_opponent(player.team, player.position.lerp(target.position, 0.55))
            self.pending_owner = interceptor
            self.pending_pass_team = None
            self.stats[player.team]["turnovers"] += 1
            ball_destination = interceptor.position
            action_text = (
                f"{TEAM_NAMES[player.team]} pass intercepted by "
                f"{TEAM_NAMES[interceptor.team]} #{interceptor.number}"
            )
        self.pending_shot_team = None
        self.pending_shot_outcome = None
        self.ball_target = pygame.Vector2(ball_destination)
        self.last_action = action_text

    def _pass_success_chance(self, player: PlayerAgent) -> float:
        defenders = self.nearby_defenders(player)
        chance = (
            player.passing_accuracy / 100.0
            + player.stamina / 360.0
            - defenders * 0.08
        )
        return max(PASS_INTERCEPT_MIN, min(PASS_INTERCEPT_MAX, chance))

    def _shoot(self, player: PlayerAgent) -> None:
        goal_x = PITCH.right + GOAL_DEPTH if player.team == "A" else PITCH.left - GOAL_DEPTH
        self.ball_owner = None
        player.has_ball = False
        self.pending_owner = None
        self.pending_pass_team = None
        self.pending_shot_team = player.team
        self.stats[player.team]["shots"] += 1
        team_name = TEAM_NAMES[player.team]

        if self._ready_to_score(player.team):
            goal_y = HEIGHT / 2 + random.uniform(-GOAL_MOUTH_HALF_HEIGHT + 10, GOAL_MOUTH_HALF_HEIGHT - 10)
            self.pending_shot_outcome = "goal"
            self.ball_target = pygame.Vector2(goal_x, goal_y)
            self.last_action = f"{team_name} shoots after required passes"
            return

        outcome = random.choices(["wide", "blocked"], weights=[0.58, 0.42], k=1)[0]
        self.pending_shot_outcome = outcome

        if outcome == "wide":
            self.stats[player.team]["wide"] += 1
            miss_side = random.choice([-1, 1])
            goal_y = HEIGHT / 2 + miss_side * random.uniform(GOAL_MOUTH_HALF_HEIGHT + 22, GOAL_MOUTH_HALF_HEIGHT + 92)
            self.ball_target = pygame.Vector2(goal_x, goal_y)
            needed = self.match_setup.passes_per_goal(player.team)
            done = self.completed_passes[player.team]
            self.last_action = f"{team_name} shoots early and sends it wide ({done}/{needed})"
        else:
            defender = self._nearest_opponent(player.team, player.position)
            self.pending_owner = defender
            self.ball_target = pygame.Vector2(defender.position)
            self.last_action = f"{team_name} early shot is blocked by {TEAM_NAMES[defender.team]} #{defender.number}"

    def _finish_shot(self) -> None:
        scorer_team = self.pending_shot_team or ("A" if self.ball_position.x > WIDTH / 2 else "B")
        shot_outcome = self.pending_shot_outcome
        self.pending_shot_team = None
        self.pending_shot_outcome = None

        if shot_outcome == "goal" and self._ready_to_score(scorer_team):
            self._add_goal(scorer_team)
            self.completed_passes[scorer_team] = 0
            team_name = TEAM_NAMES[scorer_team]
            self.goal_message = f"{team_name} team scored"
            self.goal_scoreline = f"{self.score_a} - {self.score_b}"
            self.last_action = f"{team_name} goal after required passes"
            self.flash_timer = 1.4
            self._reset_after_goal(scorer_team)
            self._finish_match_if_ready()
            return

        team_name = TEAM_NAMES[scorer_team]
        needed = self.match_setup.passes_per_goal(scorer_team)
        done = self.completed_passes[scorer_team]
        if shot_outcome == "wide":
            self.last_action = f"{team_name} shot goes wide before required passes ({done}/{needed})"
        elif self._team_score(scorer_team) >= self.match_setup.target_score(scorer_team):
            self.last_action = f"{team_name} shot saved; target score reached"
        else:
            self.last_action = f"{team_name} shot missed before {needed} passes ({done}/{needed})"
        defenders = [player for player in self.players if player.team != scorer_team]
        keeper = min(defenders, key=lambda player: abs(player.position.y - HEIGHT / 2))
        self._set_owner(keeper)

    def _reset_after_goal(self, scorer_team: str) -> None:
        for player in self.players:
            player.position = pygame.Vector2(player.home)
            player.has_ball = False
        self.ball_position = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
        self.ball_target = None
        self.pending_owner = None
        self.pending_pass_team = None
        self.pending_shot_team = None
        self.pending_shot_outcome = None
        restart_players = [player for player in self.players if player.team != scorer_team]
        self._set_owner(random.choice(restart_players))

    def _nearest_opponent(self, team: str, position: pygame.Vector2) -> PlayerAgent:
        opponents = [player for player in self.players if player.team != team]
        return min(opponents, key=lambda player: player.distance_to(position))

    def draw(self, screen: pygame.Surface, font: pygame.font.Font, small: pygame.font.Font) -> None:
        self._draw_pitch(screen)
        self._draw_players(screen, font)
        pygame.draw.circle(screen, BALL, self.ball_position, 9)
        pygame.draw.circle(screen, INK, self.ball_position, 9, 2)
        self._draw_hud(screen, font, small)
        self._draw_training_proof(screen, small)
        self._draw_goal_banner(screen, font, small)
        self._draw_full_time_banner(screen, font, small)

    def _draw_pitch(self, screen: pygame.Surface) -> None:
        screen.fill((20, 32, 35))
        pygame.draw.rect(screen, FIELD_GREEN, PITCH)
        stripe_width = PITCH.width // 10
        for index in range(10):
            if index % 2 == 0:
                rect = pygame.Rect(PITCH.left + index * stripe_width, PITCH.top, stripe_width, PITCH.height)
                pygame.draw.rect(screen, FIELD_DARK, rect)

        pygame.draw.rect(screen, LINE, PITCH, 4)
        pygame.draw.line(screen, LINE, (WIDTH // 2, PITCH.top), (WIDTH // 2, PITCH.bottom), 3)
        pygame.draw.circle(screen, LINE, (WIDTH // 2, HEIGHT // 2), 82, 3)
        pygame.draw.circle(screen, LINE, (WIDTH // 2, HEIGHT // 2), 5)

        left_box = pygame.Rect(PITCH.left, HEIGHT // 2 - 115, 150, 230)
        right_box = pygame.Rect(PITCH.right - 150, HEIGHT // 2 - 115, 150, 230)
        pygame.draw.rect(screen, LINE, left_box, 3)
        pygame.draw.rect(screen, LINE, right_box, 3)
        pygame.draw.rect(screen, LINE, (PITCH.left - GOAL_DEPTH, HEIGHT // 2 - 58, GOAL_DEPTH, 116), 3)
        pygame.draw.rect(screen, LINE, (PITCH.right, HEIGHT // 2 - 58, GOAL_DEPTH, 116), 3)

    def _draw_players(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        for player in self.players:
            pygame.draw.circle(screen, (11, 18, 24), player.position + pygame.Vector2(3, 4), 18)
            pygame.draw.circle(screen, player.color, player.position, 18)
            if player.has_ball:
                pygame.draw.circle(screen, ACCENT, player.position, 24, 3)
            label = font.render(str(player.number), True, (255, 255, 255))
            screen.blit(label, label.get_rect(center=player.position))

            stamina_width = 34
            bar = pygame.Rect(player.position.x - 17, player.position.y + 25, stamina_width, 5)
            pygame.draw.rect(screen, (28, 34, 38), bar)
            fill = pygame.Rect(bar.left, bar.top, stamina_width * (player.stamina / 100.0), 5)
            pygame.draw.rect(screen, ACCENT, fill)

    def _draw_hud(self, screen: pygame.Surface, font: pygame.font.Font, small: pygame.font.Font) -> None:
        pygame.draw.rect(screen, PANEL, (55, 16, 990, 42), border_radius=6)
        owner = self.ball_owner
        owner_text = "Loose ball" if owner is None else f"{TEAM_NAMES[owner.team]} #{owner.number} {owner.role}"
        red_passes = f"R {self.completed_passes['A']}/{self.match_setup.red_passes_per_goal}"
        blue_passes = f"B {self.completed_passes['B']}/{self.match_setup.blue_passes_per_goal}"
        target_text = f"Target {self.match_setup.red_target_score}-{self.match_setup.blue_target_score}"
        shots = f"Shots R{self.stats['A']['shots']} B{self.stats['B']['shots']}"
        parts = [
            f"Red {self.score_a} - {self.score_b} Blue",
            target_text,
            f"Minute {self.match_minute()}",
            owner_text,
            f"Passes {red_passes} {blue_passes}",
            shots,
            f"AI P {self.last_prediction.pass_success_probability * 100:.0f}%",
            f"G {self.last_prediction.goal_probability * 100:.0f}%",
            f"F {self.last_prediction.fatigue_risk * 100:.0f}%",
            self.last_prediction.recommended_strategy,
        ]
        x = 74
        for index, part in enumerate(parts):
            color = INK if index != len(parts) - 1 else (112, 78, 16)
            surface = small.render(part, True, color)
            screen.blit(surface, (x, 28))
            x += surface.get_width() + 24

        if self.flash_timer > 0:
            text = font.render("GOAL", True, ACCENT)
            screen.blit(text, text.get_rect(center=(WIDTH / 2, 36)))

    def _draw_training_proof(self, screen: pygame.Surface, small: pygame.font.Font) -> None:
        panel = pygame.Rect(55, 638, 990, 28)
        pygame.draw.rect(screen, PANEL, panel, border_radius=6)
        proof = self.predictor.training_proof_text()
        status = f"{proof} | {self.last_action}"
        text = self._render_fit(small, status, INK, panel.width - 32)
        screen.blit(text, text.get_rect(midleft=(panel.left + 16, panel.centery)))

    def _render_fit(
        self,
        font: pygame.font.Font,
        text: str,
        color: tuple[int, int, int],
        max_width: int,
    ) -> pygame.Surface:
        if font.size(text)[0] <= max_width:
            return font.render(text, True, color)

        ellipsis = "..."
        fitted = text
        while fitted and font.size(f"{fitted}{ellipsis}")[0] > max_width:
            fitted = fitted[:-1]
        return font.render(f"{fitted.rstrip()}{ellipsis}", True, color)

    def _draw_goal_banner(self, screen: pygame.Surface, font: pygame.font.Font, small: pygame.font.Font) -> None:
        if self.flash_timer <= 0:
            return

        banner = pygame.Rect(0, 0, 300, 92)
        banner.center = (WIDTH // 2, HEIGHT // 2)
        overlay = pygame.Surface(banner.size, pygame.SRCALPHA)
        alpha = int(215 * min(1.0, self.flash_timer / 0.7))
        pygame.draw.rect(overlay, (18, 26, 32, alpha), overlay.get_rect(), border_radius=8)
        pygame.draw.rect(overlay, (*ACCENT, 255), overlay.get_rect(), 3, border_radius=8)
        screen.blit(overlay, banner)

        title = font.render("GOAL", True, ACCENT)
        scorer = small.render(self.goal_message, True, LINE)
        score = font.render(self.goal_scoreline, True, LINE)
        screen.blit(title, title.get_rect(center=(banner.centerx, banner.top + 24)))
        screen.blit(scorer, scorer.get_rect(center=(banner.centerx, banner.top + 50)))
        screen.blit(score, score.get_rect(center=(banner.centerx, banner.top + 74)))

    def _draw_full_time_banner(self, screen: pygame.Surface, font: pygame.font.Font, small: pygame.font.Font) -> None:
        if not self.match_finished:
            return

        banner = pygame.Rect(0, 0, 360, 120)
        banner.center = (WIDTH // 2, HEIGHT // 2)
        overlay = pygame.Surface(banner.size, pygame.SRCALPHA)
        pygame.draw.rect(overlay, (18, 26, 32, 230), overlay.get_rect(), border_radius=8)
        pygame.draw.rect(overlay, (*ACCENT, 255), overlay.get_rect(), 3, border_radius=8)
        screen.blit(overlay, banner)

        title = font.render("FULL TIME", True, ACCENT)
        score = font.render(f"Red {self.score_a} - {self.score_b} Blue", True, LINE)
        detail = small.render("Target score completed", True, LINE)
        screen.blit(title, title.get_rect(center=(banner.centerx, banner.top + 28)))
        screen.blit(score, score.get_rect(center=(banner.centerx, banner.top + 62)))
        screen.blit(detail, detail.get_rect(center=(banner.centerx, banner.top + 92)))


def _prompt_int(label: str, default: int, minimum: int) -> int:
    while True:
        try:
            raw_value = input(f"{label} [{default}]: ").strip()
        except EOFError:
            return default

        if not raw_value:
            return default

        try:
            value = int(raw_value)
        except ValueError:
            print("Please enter a whole number.")
            continue

        if value < minimum:
            print(f"Please enter {minimum} or more.")
            continue

        return value


def prompt_match_setup(default: MatchSetup | None = None) -> MatchSetup:
    default = default or MatchSetup()
    print("\nMatch setup")
    print("Enter the final score and passes needed before each goal.")
    print("Press Enter to keep the default shown in brackets.\n")

    red_score = _prompt_int("Red team final score", default.red_target_score, 0)
    blue_score = _prompt_int("Blue team final score", default.blue_target_score, 0)
    red_passes = _prompt_int("Red passes before 1 goal", default.red_passes_per_goal, 1)
    blue_passes = _prompt_int("Blue passes before 1 goal", default.blue_passes_per_goal, 1)

    return MatchSetup(
        red_target_score=red_score,
        blue_target_score=blue_score,
        red_passes_per_goal=red_passes,
        blue_passes_per_goal=blue_passes,
    )


def run_simulation(
    model_dir: str | Path = "models",
    rows: int = 1600,
    epochs: int = 20,
    seed: int = 7,
    match_seconds: float = DEFAULT_MATCH_SECONDS,
    match_setup: MatchSetup | None = None,
) -> None:
    match_setup = match_setup or prompt_match_setup()
    predictor = MatchPredictor.load_or_train(
        model_dir=model_dir,
        rows=rows,
        epochs=epochs,
        seed=seed,
    )

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AI Football Strategy Simulator")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 18, bold=True)
    small = pygame.font.SysFont("arial", 16)
    sim = FootballSimulation(
        predictor=predictor,
        seed=seed,
        match_seconds=match_seconds,
        match_setup=match_setup,
    )

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    sim.paused = not sim.paused
                elif event.key == pygame.K_r:
                    sim = FootballSimulation(
                        predictor=predictor,
                        seed=seed,
                        match_seconds=match_seconds,
                        match_setup=match_setup,
                    )

        sim.update(dt)
        sim.draw(screen, font, small)
        pygame.display.flip()

    pygame.quit()
