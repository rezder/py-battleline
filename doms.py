'''Battleline field domains.'''
import sys
cards = list(range(1, 71))
cardPos = list(range(0, 23))
conePos = list(range(0, 3))


def cardsToTxt(cardix):
    '''Translate card index to text.'''
    cardColors = ["Green", "Red", "Purpel", "Yellow", "Blue", "Orange"]
    tactics = [
        'Traitor', 'Deserter', 'Redeploy', 'Scout', 'Mud', 'Fog', '123', '8',
        'Darius', 'Alexander'
    ]
    txt = ''
    if cardix > 0 and cardix < 61:
        d = (cardix - 1) // 10  # integer div or div floor
        m = cardix % 10
        if m == 0:
            m = 10
        txt = cardColors[d] + str(m)
    elif cardix > 60 and cardix <= 70:
        txt = tactics[cardix - 61]
    elif cardix == 0:
        txt = "None"
    else:
        print('Ilegal cardix')
        sys.exit(1)
    return txt


def cardPosToTxt(pos):
    '''Translate card position to text.'''
    txt = ''
    if pos == 0:
        txt = 'DishBot'
    elif pos == 10:
        txt = 'DishOpp'
    elif pos == 20:
        txt = 'HandLegal'
    elif pos == 21:
        txt = 'Hand'
    elif pos == 22:
        txt = 'Deck'
    elif pos < 20:
        d, m = divmod(pos, 10)
        if d == 0:
            txt = 'Flag' + m + 'Bot'
        else:
            txt = 'Flag' + m + 'Opp'

    else:
        print('Ilegal card position')
        sys.exit(1)


def conePosToTxt(pos):
    '''Translate cone position to text.'''
    txt = ''
    if pos == 0:
        txt = 'None'
    elif pos == 1:
        txt = 'Bot'
    elif pos == 2:
        txt = 'Opp'
    else:
        print('Ilegal cone position')
        sys.exit(1)
