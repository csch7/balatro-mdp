# Decks

One file per deck: the cards (`ranks`, `n_suits`) and the round rules (`hand_size`, `hands`, `discards`).
`base: <deck>` starts from another deck file and overrides only the keys listed.
Experiment configs pick a deck with `game.deck`; the blind (`target`, `reward_mode`) is set there, not here.

Balatro decks whose Ante-1 effect is only money, consumables or Jokers (Yellow, Green, Magic, Nebula, Ghost,
Zodiac) play like `standard` in this model. Not supported yet: Checkered (duplicate cards) and Erratic
(random deck), because the deck is a set of distinct (rank, suit) cards, and Plasma, which changes scoring.
