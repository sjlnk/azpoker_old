import re
import argparse
import logging
import pickle
import sys
import azlib as az
import azlib.azlogging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from datetime import datetime, timedelta


class PacificParser:

    def __init__(self):

        self.re_ai_before_turn = re.compile(r"[*][*] dealing river [*][*].*\[.*].*\n[*][*] summary [*][*]",
                                            re.IGNORECASE)
        self.re_findbets = re.compile(r'\[\$[0-9.]+\]')
        self.re_blinds_1 = re.compile(r'[*]{5}\n\$[0-9.]+/\$[0-9.]+ Blinds ')
        self.re_dollar_amount = re.compile(r'\$[0-9.]+')
        self.re_game_1 = re.compile(r'[*]{5}\n\$[0-9.]+/\$[0-9.]+ Blinds .* - [*]{3}')
        self.re_game_2 = re.compile(r'Blinds .* -')
        # self.re_collected = re.compile(r'.* collected \[ \$[0-9.]+ \]')
        self.re_sd_players = re.compile(r'.* (shows|mucks) \[')
        self.re_stack = re.compile(r'Seat [0-9]+: .* \( \$[0-9.]+ \)')
        self.re_river = re.compile(r'Dealing river [*][*].*[*][*] Summary', re.DOTALL)
        self.re_gamenumber = re.compile(r'[*]{5} 888poker Hand History for Game [0-9]+ [*]{5}')

        self.games = {
            "Pot Limit Omaha": "PLO",
            "No Limit Omaha": "NLO",
            "No Limit Holdem": "NLHE"
        }

    def allin_before_river(self, data):
        if data is str:
            if self.re_ai_before_turn.search(data):
                return True
            return False
        else:
            res = []
            for h in data:
                if self.re_ai_before_turn.search(h):
                    res.append(True)
                else:
                    res.append(False)
            return np.asarray(res)

    def _player_allin_before_river_single(self, hand, player):
        hero_in_sd = False
        for m in self.re_sd_players.finditer(hand):
            s = hand[m.start():m.end()]
            if player in s:
                hero_in_sd = True
                break
        if not hero_in_sd:
            return False
        m = self.re_river.search(hand)
        river = hand[m.start():m.end()]
        riveractions = river.splitlines()[1:-1]
        for s in riveractions:
            if player in s:
                return False
        # if riveractions:
            # print(hand)
        return True

    def player_allin_before_river(self, data, player):
        if data is str:
            return self._player_allin_before_river_single(data, player)
        else:
            res = []
            for h in data:
                res.append(self._player_allin_before_river_single(h, player))
            return np.asarray(res)

    def _pot_for_player_single(self, hand, player):
        players_sharing = []
        for m in self.re_sd_players.finditer(hand):
            s = hand[m.start():m.end()]
            players_sharing.append(s.split()[0])
        if player not in players_sharing:
            return np.nan
        stack_strings = self.re_stack.findall(hand)
        stacks = []
        herostack = None
        for s in stack_strings:
            for plr in players_sharing:
                if plr in s:
                    stack = float(self.re_dollar_amount.findall(s)[0][1:])
                    if not herostack and player in s:
                        herostack = stack
                    else:
                        stacks.append(stack)
        pot = 0
        for f in stacks:
            pot += min(herostack, f)
        pot += min(herostack, max(stacks))
        return pot

    def pot_for_player(self, data, player):
        if data is str:
            return self._pot_for_player_single(data, player)
        else:
            res = []
            for h in data:
                res.append(self._pot_for_player_single(h, player))
            return np.asarray(res)

    def total_pot_before_rake(self, data):
        if data is str:
            bsum = 0.0
            for s in self.re_findbets.findall(data):
                bsum += float(s[2:-1])
            return bsum
        else:
            res = []
            for h in data:
                bsum = 0.0
                for s in self.re_findbets.findall(h):
                    bsum += float(s[2:-1])
                res.append(bsum)
            return np.asarray(res)

    @staticmethod
    def player_in_sd(data, player):
        regex = re.compile(r'{} shows \[.*\]'.format(player))
        if data is str:
            if regex.search(data):
                return True
            return False
        else:
            res = []
            for h in data:
                if regex.search(h):
                    res.append(True)
                else:
                    res.append(False)
            return np.asarray(res)

    @staticmethod
    def hero_is_playing(data):
        if data is str:
            if "Dealt to " in data:
                return True
            return False
        else:
            res = []
            for h in data:
                if "Dealt to " in h:
                    res.append(True)
                else:
                    res.append(False)
            return np.asarray(res)

    def get_blinds(self, data):
        if data is str:
            s = self.re_blinds_1.search(data)
            s = self.re_dollar_amount.findall(s)
            return float(s[0][1:]), float(s[1][1:])
        else:
            res = []
            for h in data:
                s = self.re_blinds_1.search(h)
                s = self.re_dollar_amount.findall(h[s.start():s.end()])
                res.append((float(s[0][1:]), float(s[1][1:])))
            return np.asarray(res)

    def get_game(self, data):
        if data is str:
            sgame = self.re_game_2.findall(self.re_game_1.findall(data)[0])[0][7:-2]
            return self.games[sgame]
        else:
            res = []
            for h in data:
                sgame = self.re_game_2.findall(self.re_game_1.findall(h)[0])[0][7:-2]
                res.append(self.games[sgame])
            return np.asarray(res)

    @staticmethod
    def is_tournament(data):
        if data is str:
            if "Tournament #" in data:
                return True
            return False
        else:
            res = []
            for h in data:
                if "Tournament #" in h:
                    res.append(True)
                else:
                    res.append(False)
            return np.asarray(res)

    @staticmethod
    def is_real_money(data):
        if data is str:
            if "(Real Money)" in data:
                return True
            return False
        else:
            res = []
            for h in data:
                if "(Real Money)" in h:
                    res.append(True)
                else:
                    res.append(False)
            return np.asarray(res)


def print_basic_info(df):

    logging.info("{} hands total".format(len(df)))

    allin_hero = df[pp.player_allin_before_river(df.hh, args.hero)]
    logging.info("{} hands where {} went all-in on turn or earlier".format(len(allin_hero), args.hero))

    pots = pp.pot_for_player(allin_hero.hh, args.hero)
    allin_hero['pot_share'] = pots
    len_before = len(allin_hero)
    allin_hero = allin_hero[~allin_hero.pot_share.isnull()]
    nullpots_dropped = len_before - len(allin_hero)
    if nullpots_dropped > 0:
        logging.warning("Null pots dropped: {}".format(nullpots_dropped))

    logging.info("Total equity changed on all-in hands where hero was involved: {:,.0f}"
                 .format(allin_hero.pot_share.sum()))
    logging.info("Average pot share on hero's all-in hands: {:.2f}".format(allin_hero.pot_share.mean()))
    return allin_hero


def generate_randomwalk(weights, limits, nsamples):
    meanbins = limits[1:] - np.diff(limits) / 2
    cd = np.cumsum(weights)
    cd /= cd[-1]
    r = np.random.random(nsamples)
    res = []
    for f in r.flat:
        res.append(meanbins[(cd > f).argmax()])
    return np.asarray(res).reshape(r.shape)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="script to analyze hand histories")
    parser.add_argument("files", nargs='+', help="hand history files to analyze")
    parser.add_argument("hero", help="name of the hero")
    parser.add_argument("--games", nargs='+', help="choose only these games")
    parser.add_argument("--visualize", action="store_true", help="visualize data")
    parser.add_argument("--is-stakes", nargs='+', help="which stakes to accept (for example PLO100)")
    parser.add_argument("--is-pot-multiplier", type=float, help="multiply pots/stakes by given factor")
    parser.add_argument("--is-samples", type=int, default=100000, help="number of sample runs")
    parser.add_argument("--is-runlen", type=int, nargs='+', help="how many allins to simulate per run [start end inc]")
    parser.add_argument("--is-prem", nargs='+', help="premiums to simulate: [start end inc] or just one premium")
    parser.add_argument("--is-avgequity", type=float, default=0.5, help="average equity on allins")
    parser.add_argument("--is-evneutral", action="store_true", help="ev-neutral insurance")
    parser.add_argument("--is-target-revenue", type=float, help="target revenue per sample (single-setting only)")
    parser.add_argument("--is-pickle-res", nargs='?', const="res.p", help="pickle result from mc simulation")

    simgroup = parser.add_mutually_exclusive_group()
    simgroup.add_argument("--split-stakes", action="store_true", help="split results by stakes/games")
    simgroup.add_argument("--sim-multisetting", action="store_true", help="run multi-parameter insurance simulation")
    simgroup.add_argument("--sim-singlesetting", action="store_true",
                          help="run insurance MC test with one set of params")

    vgroup = parser.add_mutually_exclusive_group()
    vgroup.add_argument("-v", action="count", default=0, help="verbosity")
    vgroup.add_argument("--quiet", action="store_true", help="disable stdout-output")

    args = parser.parse_args()

    az.azlogging.quick_config(args.v, args.quiet, fmt="")

    all_hands = []
    all_gameno = []
    all_site = []

    # re_pacific_gameno_2 = re.compile(r'[0-9]*')

    for file in args.files:

        with open(file, 'r') as hhfile:
            hand = hhfile.read()
            # 888 Poker
            if re.search(r'[*]{5} 888poker Hand History for Game [0-9]+ [*]{5}', hand):
                hands = hand.split('#Game No : ')
                for h in hands:
                    try:
                        firstln = h.find('\n')
                        all_gameno.append(int(h[:firstln]))
                    except ValueError:
                        if h:
                            logging.debug("Cannot parse hand: \n{}".format(h))
                        continue
                    all_hands.append(h[firstln+1:])
                    all_site.append(0)

    # all_hands
    df = DataFrame({'hh': all_hands, 'site': all_site}, index=all_gameno)
    len_before = len(df)
    logging.info("{} hands imported.".format(len_before))

    df = df.groupby(level=0).first()
    num_dupes = len_before - len(df)
    if num_dupes > 0:
        logging.info("{} duplicates dropped.".format(num_dupes))

    pp = PacificParser()

    blinds = pp.get_blinds(df.hh)
    sb, bb = zip(*blinds)
    df['sb'] = sb
    df['bb'] = bb

    df['game'] = pp.get_game(df.hh)

    len_before = len(df)
    df = df[~pp.is_tournament(df.hh)]
    num_tournaments = len_before - len(df)
    if num_tournaments > 0:
        logging.debug("{} tournament hands dropped.".format(num_tournaments))

    len_before = len(df)
    df = df[pp.is_real_money(df.hh)]
    num_play_money = len_before - len(df)
    if num_play_money > 0:
        logging.debug("{} play money hands dropped.".format(num_play_money))

    len_before = len(df)
    df = df[pp.hero_is_playing(df.hh)]
    num_not_active = len_before - len(df)
    if num_not_active > 0:
        logging.debug("{} non-active hands dropped.".format(num_not_active))

    if args.sim_multisetting or args.sim_singlesetting:
        df['gametag'] = df.game + (df.bb * 100).astype(int).astype('str')
        if args.is_stakes:
            len_before = len(df)
            df = df[df.gametag.isin(args.is_stakes)]
            num_games_filtered_out = len_before - len(df)
            if num_games_filtered_out > 0:
                logging.debug("{} hands filtered out due to game selection.".format(num_games_filtered_out))

        len_filtered = len(df)
        logging.info("Hands after filtering: {}".format(len_filtered))

        if len_filtered == 0:
            logging.info("No hands remaining after filtering process.")
            sys.exit(0)

        allin_hero = print_basic_info(df)
        if args.is_pot_multiplier:
            allin_hero[['bb', 'pot_share']] *= args.is_pot_multiplier
            logging.info("Multiplied pots and stakes by {}".format(args.is_pot_multiplier))

        weights, limits, _ = plt.hist(allin_hero.pot_share.values, bins=100)
        if args.visualize:
            plt.show()
    else:
        if args.games:
            len_before = len(df)
            df = df[df.game.isin(args.games)]
            num_games_filtered_out = len_before - len(df)
            if num_games_filtered_out > 0:
                logging.debug("{} hands filtered out due to game selection.".format(num_games_filtered_out))

        len_filtered = len(df)
        logging.info("Hands after filtering: {}".format(len_filtered))

        if args.split_stakes:
            sdict = df.groupby(['game', 'bb']).groups
            for stake, data in sdict.items():
                gamestr = stake[0] + str(int(stake[1] * 100))
                logging.info("\n{}:".format(gamestr))
                sdf = df.loc[data]
                print_basic_info(sdf)

            logging.info("\nTOTAL:")
            allin_hero = print_basic_info(df)
        else:
            allin_hero = print_basic_info(df)

    if args.sim_multisetting:

        if args.is_runlen:
            if len(args.is_runlen) == 3:
                runlens = [i for i in range(args.is_runlen[0], args.is_runlen[1], args.is_runlen[2])]
            elif len(args.is_runlen) == 1:
                runlens = args.is_runlen
            else:
                logging.critical("--is-runlen parameter needs exactly 1 or 3 inputs, {} was provided"
                                 .format(len(args.is_runlen)))
                sys.exit(2)
        else:
            runlens = [i for i in range(200, 2001, 200)]

        if args.is_prem:
            args.is_prem[0] = float(args.is_prem[0])
            args.is_prem[1] = float(args.is_prem[1])
            args.is_prem[2] = int(args.is_prem[2])
            prems = np.linspace(args.is_prem[0], args.is_prem[1], args.is_prem[2])
        else:
            prems = [i / 100 for i in range(11)]

        # generate_randomwalk is slow so we need to use it sparingly!
        pot_collections = generate_randomwalk(weights, limits, (runlens[-1], 100))
        all_pots = np.zeros((runlens[-1], args.is_samples))
        randints = np.random.randint(pot_collections.shape[1], size=args.is_samples)
        for i in range(args.is_samples):
            all_pots[:, i] = pot_collections[:, randints[i]]

        mean_pot = np.mean(all_pots)
        mean_bb = allin_hero.bb.mean()
        revenue_per_hand = mean_pot * (len(allin_hero) / len_filtered)
        logging.info("Average pot size on generated pots: {}, Average BB: {}".format(mean_pot, mean_bb))

        for runlen in runlens:
            potsizes = all_pots[:runlen, :]
            revenues = np.sum(potsizes, axis=0)
            act_mean_revenue = np.mean(revenues)
            bm_mean_revenue = mean_pot * runlen
            estimated_hands = bm_mean_revenue / revenue_per_hand
            logging.info("\nPots: {} Mean-revenue: {:,.0f} Actual-mean-revenue: {:,.0f} Est.Hands: {:,.0f}"
                         .format(runlen, bm_mean_revenue, act_mean_revenue, estimated_hands))
            for prem in prems:
                samples = np.random.random((runlen, args.is_samples))
                rwalks = np.where(samples < args.is_avgequity, -1, 1)
                moneywise = rwalks * potsizes
                result = np.cumsum(moneywise, axis=0)
                rwalk_mean = np.mean(rwalks)
                mwise_mean = rwalk_mean * mean_pot
                wholerun_ev = mwise_mean * runlen
                endres = result[-1, :]
                endres -= wholerun_ev
                logging.debug("rwalk-mean: {:g} moneywise-mean: {:g} whole-run ev: {:g} res evadj mean: {:g}"
                              .format(rwalk_mean, mwise_mean, wholerun_ev, np.mean(endres)))
                if not args.is_evneutral:
                    endres = np.where(endres > 0, 0, endres)
                mean_endres_no_prem = np.mean(endres)
                endres += revenues * prem
                prem_moneywise = act_mean_revenue * prem
                if args.is_evneutral:
                    est_bb100 = (-prem_moneywise) / (mean_bb * (estimated_hands / 100))
                else:
                    est_bb100 = (-prem_moneywise - mean_endres_no_prem) / (mean_bb * (estimated_hands / 100))
                winpct = np.sum(endres >= 0) / len(endres)
                logstr = "Premium={:g} Premium$={:.0f} BB/100={:.2f} mean={:g}"\
                    .format(prem, prem_moneywise, est_bb100, np.mean(endres))
                logstr += "std={:g} win%={:.2%} 95%={:g} 99%={:g} min={:g}"\
                    .format(np.std(endres), winpct, np.percentile(endres, 5), np.percentile(endres, 1), np.min(endres))
                if args.is_evneutral:
                    logstr += " max={:g}".format(np.max(endres))
                logging.info(logstr)

    elif args.sim_singlesetting:

        if args.is_runlen:
            if len(args.is_runlen) == 3:
                runlens = [i for i in range(args.is_runlen[0], args.is_runlen[1], args.is_runlen[2])]
            elif len(args.is_runlen) == 1:
                runlens = args.is_runlen
            else:
                logging.critical("--is-runlen parameter needs exactly 1 or 3 inputs, {} was provided"
                                 .format(len(args.is_runlen)))
                sys.exit(2)
        else:
            runlens = [i for i in range(2, 21, 2)]

        mean_pot = allin_hero.pot_share.mean()
        max_pots_required = round((args.is_target_revenue / mean_pot) * 2)

        # generate_randomwalk is slow so we need to use it sparingly!
        pot_collections = generate_randomwalk(weights, limits, (max_pots_required, 100))
        max_pot_idx = 0
        for i in range(100):
            csum = np.cumsum(pot_collections[:, i])
            lastidx = (csum > args.is_target_revenue).argmax() + 1
            max_pot_idx = max(max_pot_idx, lastidx)
            pot_collections[lastidx:, i] = np.nan
        pot_collections = pot_collections[:max_pot_idx, :]

        all_pots = np.zeros((max_pot_idx, args.is_samples))
        randints = np.random.randint(pot_collections.shape[1], size=args.is_samples)
        for i in range(args.is_samples):
            all_pots[:, i] = pot_collections[:, randints[i]]

        all_pots = np.ma.masked_invalid(all_pots)
        mean_pot = np.nanmean(pot_collections)
        mean_bb = allin_hero.bb.mean()
        revenue_per_hand = mean_pot * (len(allin_hero) / len_filtered)
        bm_mean_revenue = np.mean(np.sum(all_pots, axis=0))
        logging.info("Average pot size on generated pots: {:.2f}, Average BB: {:.3f}".format(mean_pot, mean_bb))
        estimated_hands = bm_mean_revenue / revenue_per_hand
        estimated_pots = np.mean(np.sum(~np.isnan(pot_collections), axis=0))
        logging.info("Avg hands/insurance period: {:,.0f}, Avg # of all-in pots: {:,.0f}"
                     .format(estimated_hands, estimated_pots))

        assert(len(args.is_prem) == 1)
        prem = float(args.is_prem[0])

        time_elapsed_since_info = timedelta(0)
        calc_times = []

        for runlen in runlens:
            res = np.zeros((runlen, args.is_samples))
            for i in range(runlen):
                start_time = datetime.utcnow()
                samples = np.random.random(all_pots.shape)
                rwalks = np.where(samples < args.is_avgequity, -1, 1)
                moneywise = rwalks * all_pots
                # moneywise = np.ma.masked_invalid(moneywise)
                result = moneywise.cumsum(axis=0)
                rwalk_mean = np.mean(rwalks)
                mwise_mean = rwalk_mean * mean_pot
                wholerun_ev = mwise_mean * estimated_pots
                endres = []
                for i2 in range(result.shape[1]):
                    endres.append(result[~result.mask[:, i2], i2][-1])
                endres = np.asarray(endres)
                if not az.feqd(0.5, args.is_avgequity):
                    endres -= wholerun_ev
                    logging.debug("rwalk-mean: {:g} moneywise-mean: {:g} whole-run ev: {:g} res evadj mean: {:g}"
                                  .format(rwalk_mean, mwise_mean, wholerun_ev, np.mean(endres)))
                else:
                    logging.debug("rwalk-mean: {:g} moneywise-mean: {:g} whole-run ev: {:g}"
                                  .format(rwalk_mean, mwise_mean, wholerun_ev))
                if not args.is_evneutral:
                    endres = np.where(endres > 0, 0, endres)
                endres += args.is_target_revenue * prem
                res[i, :] = endres
                time_elapsed = datetime.utcnow() - start_time
                time_elapsed_since_info += time_elapsed
                calc_times.append(time_elapsed)
                if len(runlens) == 1 and time_elapsed_since_info > timedelta(seconds=10):
                    meantime = np.mean(calc_times)
                    samples_per_sec = 1 / meantime.total_seconds()
                    est_time_remaining = meantime * (runlen - len(calc_times))
                    logging.info("samples calculcated: {}, samples/sec: {:g}, est. time remaining: {}"
                                 .format(len(calc_times), samples_per_sec, est_time_remaining))
                    time_elapsed_since_info = timedelta(0)

            logging.info("{} periods, total revenue={:,.0f}, total premium={:,.0f}, est.hands={:,.0f}"
                         .format(runlen, args.is_target_revenue * runlen, args.is_target_revenue * prem * runlen,
                                 estimated_hands * runlen))

            results = res.sum(axis=0)
            winpct = np.sum(results >= 0) / len(results)
            logstr = "mean={:g} std={:g} max={:g} min={:g} winpct={:g} 95%={:g} 99%={:g}"\
                .format(np.mean(results), np.std(results), np.max(results), np.min(results), winpct,
                        np.percentile(results, 5), np.percentile(results, 1))
            logging.info(logstr)

        if args.is_pickle_res:
            pickle.dump(res, open(args.is_pickle_res, 'wb'))



#
#
#
# for i1 in range(100, 1000, 100):
#     for i2 in range(11):
#         prem = i2 * 0.01
#         samples = np.random.random((i1,100000))
#         rwalks = np.where(samples < 0.4, -1, 1)
#         result = np.cumsum(rwalks, axis=0)
#         endres = result[-1,:]
#         mean_res = np.mean(rwalks) * samples.shape[0]
#         endres = endres - mean_res
#         endres = np.where(endres > 0, 0, endres)
#         endres += samples.shape[0] * prem
#         print("Pots={}, Premium={}, mean={:g}, std={:g}, 95%={:g}, min={:g}".format(i1, prem, mean(endres), std(endres), np.percentile(endres, 5), min(endres)))
#
#
#
#
