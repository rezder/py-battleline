'''Viewer for battleline machine data csv files'''
import sys
import csv
import doms


def main(file):
    flds = createFlds()
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            y = int(row[0])  # y
            x = []
            fldix = 0
            ix = 1
            for rowix, cell in enumerate(row):
                if rowix != 0:
                    l = cell.split(':')
                    i = int(l[0])
                    v = float(l[1])
                    fld = flds[fldix]
                    while hasattr(fld, 'scale') and ix != i:
                        fldix = fldix + 1
                        fld = flds[fldix]
                        ix = ix + 1
                        x.append(0)
                    try:
                        span = len(fld.values)
                        print(fld.values)
                        print(fld.name, span, i, ix, i - ix)
                        x.append(fld.values[i - ix])
                        ix = ix + span
                    except AttributeError:
                        if i == ix:
                            x.append(int(round(v * fld.scale, 0)))
                        ix = ix + 1
                    fldix = fldix + 1

            printShort(y, x)


def printShort(y, x):
    '''Print a short version of battleline game position.

    Args:
    y: lable 1 or 0.
    x: mpos battleline machine data.
    '''
    print('---------- Position ----------')
    if y == 1:
        print('Move: True')
    else:
        print('Move: False')

    print('Hand: {}'.format(printCards(handCards(x[:70]))))

    for i in range(1, 10):
        f = Flag(x[:70], x[70:80], i)
        print(f)

    dishBot, dishOpp = dishCards(x[:70])
    print('Dish {},{}'.format(printCards(dishBot), printCards(dishOpp)))
    print('Move: {}'.format(printMove(x[len(x) - 4:])))
    print('-------- End Position --------')


def printMove(mMove):
    '''Print a battleline machine move.'''
    if mMove[0] != 0:
        txt = ''
        txt = txt + doms.cardsToTxt(mMove[0])
        txt = txt + ' to '
        txt = txt + doms.cardPosToTxt(mMove[1])
        if mMove[2] != 0:
            txt = txt + ' and '
            txt = txt + doms.cardsToTxt(mMove[2])
            txt = txt + ' to '
            txt = txt + doms.cardPosToTxt(mMove[3])

    else:
        txt = 'Pass'

    return txt


class Flag:
    '''A battleline flag.'''

    def __init__(self, cards, cones, flagNo):
        '''Init a battleline flag.

        Args:
        cards: All card positions.
        cones: All cone positions
        flagNo: The flag number(not index).
        '''
        self.cardsBot = []
        self.cardsOpp = []
        self.conePos = cones[flagNo - 1]
        self.flagNo = flagNo
        for ix, cardPos in enumerate(cards):
            if cardPos == flagNo:
                self.cardsBot.append(ix + 1)
            elif cardPos == flagNo + 10:
                self.cardsOpp.append(ix + 1)

    def __str__(self):
        txt = "Flag {0}: {1} {2} {3} {4} {5}"
        coneTxt = ('x', 'x', 'x')
        if self.conePos == 1:
            coneTxt = ('x', '_', '_')
        elif self.conePos == 0:
            coneTxt = ('_', 'x', '_')
        else:
            coneTxt = ('_', '_', 'x')

        return txt.format(self.flagNo, coneTxt[0],
                          printCards(self.cardsBot), coneTxt[1],
                          printCards(self.cardsOpp), coneTxt[2])


def printCards(cardixs):
    '''Print a list of cardixs.'''
    if len(cardixs) > 0:
        s = '['
        for cardix in cardixs:
            s = s + doms.cardsToTxt(cardix)
            s = s + ','
        s = s[:len(s) - 1]
        s = s + ']'
    else:
        s = '[]'

    return s


def dishCards(cards):
    '''Create the dish cardix lists.

    Args:
    cards: All card positions.

    Returns:
    dishBot: The dished cardixs by the bot.
    dishOpp: The dished cardix by the opponent.
    '''
    dishOpp = []
    dishBot = []
    for ix, cardPos in enumerate(cards):
        if cardPos == 0:
            dishBot.append(ix + 1)
        elif cardPos == 10:
            dishOpp.append(ix + 1)

    return dishBot, dishOpp


def deckCards(cards):
    '''Create the deck lists.

    Args:
    cards: All the card postions.

    Returns:
    deckTroop: The cardixs of the troop deck.
    deckTac: The cardixs of the tactic deck.
    '''
    deckTac = []
    deckTroop = []
    for ix, cardPos in enumerate(cards):
        if cardPos == 22:
            if ix + 1 > 60:
                deckTac.append(ix + 1)
            else:
                deckTroop.append(ix + 1)

    return deckTroop, deckTac


def handCards(cards):
    '''Returns the hand from all the all the cards positions.'''
    hand = []
    for ix, cardPos in enumerate(cards):
        if cardPos == 20 or cardPos == 21:
            hand.append(ix + 1)

    return hand


def createFlds():
    '''Returns all the machine battleline feature fields.'''
    posFlds = []
    guiles = range(61, 65)
    guilePos = [22, 0, 10, 21, 20]

    for cardix in range(1, 71):
        if cardix in guiles:
            posFlds.append(CardPosFld('Card' + str(cardix), guilePos, []))
        else:
            posFlds.append(CardPosFld('Card' + str(cardix), doms.cardPos, []))

    for flagNo in range(1, 10):
        posFlds.append(ConePosFld('Flag' + str(flagNo)))

    posFlds.append(ValueFld('Opp_Hand_No_Troops', 7))
    posFlds.append(ValueFld('Opp_Hand_No_Tactics', 4))
    posFlds.append(CardFld('Scout_Return_Bot_First_Card', [], [64]))
    posFlds.append(CardFld('Scout_Return_Bot_Second_Card', [], [64]))
    posFlds.append(ValueFld('Opp_Know_Deck_Tactics', 2))
    posFlds.append(ValueFld('Opp_Know_Deck_Troops', 2))
    posFlds.append(CardFld('Opp_Know_Card_On_Hand1', [], [64]))
    posFlds.append(CardFld('Opp_Know_Card_On_Hand2', [], [64]))
    posFlds.append(ValueFld('Pass_Possible', 1))
    posFlds.append(CardFld('Move_First_Card', [], []))
    moveFirstPos = range(0, 10)
    posFlds.append(CardPosFld('Move_First_Card_Pos', moveFirstPos, []))
    posFlds.append(CardFld('Move_Second_Card', [], guiles))
    moveSecondPos = range(0, 11)
    posFlds.append(CardPosFld('Move_Second_Card_Pos', moveSecondPos, []))

    return posFlds


class ValueFld:
    '''A value field.

    A field that contain a value.
    '''

    def __init__(self, name, scale):
        self.name = name
        self.scale = scale

    def valueToTxt(self, value):
        '''Translate value to text.'''
        return str(value)


class CardFld:
    '''A field that contains card indexes.

    The field contain a subset of all the possible card indexes ind the battleline game.
    '''

    def __init__(self, name, posList, negList):
        '''Init the field.

        The field are constructed from a positiv list or a negative
        list. The positive list is just used as the possible value the field
        can hold if different from empty the negative list is ignored. If
        the negative is used the values are subtracted from all the possible
        values.

        Args:
        name: The name of the field,
        posList: The possitive list of card indexes.
        negList: The negative list of card indexes.
        '''
        self.name = name
        self.values = [0]
        if len(posList) > 0:
            self.values.extend(posList)
        else:
            self.values.extend([x for x in doms.cards if x not in negList])

    def valueToTxt(self, cardix):
        '''Translate a card index to text.'''
        if cardix not in self.values:
            print('Ilegal cardix')
            sys.exit(1)

        txt = 'Empty'
        if cardix != 0:
            txt = doms.cardsToTxt(cardix)

        return txt


class CardPosFld:
    '''A card position field.'''

    def __init__(self, name, posList, negList):
        """Init a card postion field.

        The field are constructed from a positiv list or a negative
        list. The positive list is just used as the possible values the field
        can hold if different from empty the negative list is ignored. If
        the negative is used the values are subtracted from all the possible
        values.

        Args:
        name: The name of the field,
        posList: The possitive list of card postions.
        negList: The negative list of card positions.
        """

        self.name = name
        if len(posList) > 0:
            self.values = posList
        else:
            self.values = [x for x in doms.cardPos if x not in negList]

    def valueToTxt(self, posix):
        """Translate a card postion value to text."""
        if posix not in self.values:
            print('Ilegal card postion index')
            sys.exit(1)

        return doms.cardPosToTxt(posix)


class ConePosFld:
    """A cone postion field."""

    def __init__(self, name):
        self.name = name
        self.values = doms.conePos

    def valueToTxt(self, posix):
        """Translate a cone position value to text."""
        if posix not in self.values:
            print('Ilegal card postion index')
            sys.exit(1)

        return doms.conePosToTxt(posix)


if __name__ == "__main__":
    main(sys.argv[1])
