# Candidate 15F / 15M TEDLIUM pool (pre-code-change)

This is a concrete candidate pool for expanding the gender-transfer experiment
without changing the whole-talk adaptation protocol.

## Selection rule currently under consideration
- Keep the existing dev/test evaluation speakers, even if their max transcript gap is >5s.
- For **train** additions, require a whole-talk max transcript gap below the smallest value that gives us enough verified female speakers.

## Key finding
A strict train threshold of **5s** is nowhere near enough.

After fixing the STM audit to sort utterances by timestamp before measuring
consecutive gaps, the female side becomes much more reasonable: to reach
**15 female speakers total** while keeping the existing 3 female dev/test
speakers, the current verified shortlist needs roughly **max gap <= 45.43s**
(using the 12 train female candidates below).

That is still a relaxation from 5s, but it is no longer the obviously bogus
multi-minute gap estimate from the earlier broken audit.

## Female set (15 total)

### Existing dev/test females (whitelisted)
- `AimeeMullins_2009P` — split=`test` — max_gap=`8.17s`
- `JaneMcGonigal_2010` — split=`test` — max_gap=`2.11s`
- `ElizabethGilbert_2009` — split=`dev` — max_gap=`3.89s`

### Train female additions (12)
These all have TED speaker pages that were fetched as an online identity check.

1. `JoAnnKucheraMorin_2009` — max_gap=`8.05s`  
   Source: <https://www.ted.com/speakers/joann_kuchera_morin>
2. `MarisaFickJordan_2007G2` — max_gap=`14.42s`  
   Source: <https://www.ted.com/speakers/marisa_fick_jordan>
3. `CarolinePhillips_2010G` — max_gap=`11.72s`  
   Source: <https://www.ted.com/speakers/caroline_phillips>
4. `DiannaCohen_2010Z` — max_gap=`14.93s`  
   Source: <https://www.ted.com/speakers/dianna_cohen>
5. `AlisonJackson_2005G` — max_gap=`23.91s`  
   Source: <https://www.ted.com/speakers/alison_jackson>
6. `LakshmiPratury_2007` — max_gap=`12.12s`  
   Source: <https://www.ted.com/speakers/lakshmi_pratury>
7. `CarolynPorco_2009U` — max_gap=`12.04s`  
   Source: <https://www.ted.com/speakers/carolyn_porco>
8. `JaneChen_2009I` — max_gap=`14.88s`  
   Source: <https://www.ted.com/speakers/jane_chen>
9. `RachelPike_2009G` — max_gap=`45.43s`  
   Source: <https://www.ted.com/speakers/rachel_pike>
10. `StaceyKramer_2010` — max_gap=`15.92s`  
    Source: <https://www.ted.com/speakers/stacey_kramer>
11. `LauraTrice_2008` — max_gap=`16.04s`  
    Source: <https://www.ted.com/speakers/laura_trice>
12. `JessaGamble_2010GU` — max_gap=`16.80s`  
    Source: <https://www.ted.com/speakers/jessa_gamble>

## Male set (15 total)
This can be satisfied using dev/test recordings alone; no train males are required.
One possible 15-speaker set is:

1. `BrianCox_2009U` — split=`dev` — max_gap=`0.00s`
2. `WadeDavis_2003` — split=`dev` — max_gap=`1.67s`
3. `GaryFlake_2010` — split=`test` — max_gap=`3.13s`
4. `JamesCameron_2010` — split=`test` — max_gap=`4.15s`
5. `MichaelSpecter_2010` — split=`test` — max_gap=`4.26s`
6. `DanielKahneman_2010` — split=`test` — max_gap=`4.54s`
7. `CraigVenter_2008` — split=`dev` — max_gap=`4.54s`
8. `TomWujec_2010U` — split=`test` — max_gap=`4.68s`
9. `DavidMerrill_2009` — split=`dev` — max_gap=`7.29s`
10. `BarrySchwartz_2005G` — split=`dev` — max_gap=`9.36s`
11. `DanBarber_2010` — split=`test` — max_gap=`10.30s`
12. `BlaiseAguerayArcas_2007` — split=`dev` — max_gap=`11.49s`
13. `BillGates_2010` — split=`test` — max_gap=`13.99s`
14. `AlGore_2009` — split=`dev` — max_gap=`14.81s`
15. `RobertGupta_2010U` — split=`test` — max_gap=`20.47s`

Alternate male holdout: `EricMead_2009P` (test, `21.18s`).

## Implication
If we stick to whole-talk adaptation and want 15 female / 15 male, the data is
not compatible with a 5s train-gap cutoff. With the corrected audit, the
current verified female shortlist points to a train cutoff in roughly the
**~45s** range, assuming the female shortlist above is acceptable.

## Next implementation step
- Encode this pool in a manifest file.
- Patch `run_cross_speaker_gender_tedlium.py` to load speakers from the manifest
  instead of the current hardcoded 6-speaker dict.
- Update the README to describe the new pool construction and the train-gap rule.
