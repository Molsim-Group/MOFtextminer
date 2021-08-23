from fuzzywuzzy import fuzz
from collections import defaultdict

from mof import MOF
from mofdict import MofDictionary
from utils import transform_unit, ComputableNamedList


type_precursor = dict


class AccuracyResult(object):
    def __init__(self, compare_type):
        self.compare_type = compare_type
        self.data = {}
        self.name_accuracy = None

    def __repr__(self):
        f1 = self.f1_score
        recall = self.recall
        precision = self.precision

        text = ""
        if self.name_accuracy is not None:
            text += f'Name accuracy : {self.name_accuracy}\n'

        for data_name in self.keys():
            text += f"{data_name} | precision : {precision[data_name]} | recall : {recall[data_name]} | f1 score : {f1[data_name]}\n"

        return text

    def __getitem__(self, item):
        return self.data[item]

    def get(self, item, default=None):
        try:
            return self.data[item]
        except (KeyError, IndexError):
            if default is None:
                return ComputableNamedList('tp fp fn')
            else:
                return default

    def items(self):
        return self.data.items()

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    @property
    def f1_score(self):
        f1_dict = {}
        precision = self.precision
        recall = self.recall
        for data_name in self.keys():
            rec = recall.get(data_name, 0)
            prec = precision.get(data_name, 0)
            try:
                f1 = 2 * prec * rec / (prec + rec)
            except (ZeroDivisionError, RuntimeWarning):
                f1 = 0
            f1_dict[data_name] = f1
        return f1_dict

    @property
    def recall(self):
        recall_dict = {}
        for data_name, data_num in self.items():
            tp = data_num.tp
            fn = data_num.fn
            try:
                recall_dict[data_name] = tp / (tp + fn)
            except (ZeroDivisionError, RuntimeWarning):
                recall_dict[data_name] = 0
        return recall_dict

    @property
    def precision(self):
        precision_dict = {}
        for data_name, data_num in self.items():
            tp = data_num.tp
            fp = data_num.fp
            try:
                precision_dict[data_name] = tp / (tp + fp)
            except (ZeroDivisionError, RuntimeWarning):
                precision_dict[data_name] = 0
        return precision_dict

    def update(self, obj):
        if self.name_accuracy is None and obj.name_accuracy is not None:
            self.name_accuracy = obj.name_accuracy

        try:
            for data_name, obj_num in obj.items():
                data_num = self[data_name]
                data_num += obj_num
        except KeyError:
            raise TypeError(f"{type(self)} can not append {type(obj)}")


class PrecursorAccuracyResult(AccuracyResult):
    def __init__(self):
        super(PrecursorAccuracyResult, self).__init__('mof')
        for data_name in ['composition']:
            self.data[data_name] = ComputableNamedList('tp fp fn')

    def compare(self, pre1, pre2, composition_threshold=0.01):
        result = _compare_precursor(pre1, pre2, composition_threshold=composition_threshold)
        self.update(result)
        return self


class MofAccuracyResult(AccuracyResult):
    def __init__(self):
        super(MofAccuracyResult, self).__init__('mof')
        for data_name in ['m_precursor', 'm_composition', 'o_precursor', 'o_composition',
                          's_precursor', 'time', 'temperature']:
            self.data[data_name] = ComputableNamedList('tp fp fn')

    def compare(self, mof1, mof2, composition_threshold=0.01, name_threshold=0.8):
        result = _compare_mof(mof1, mof2, composition_threshold, name_threshold)
        self.update(result)
        return self


class MofdictionaryAccuracyResult(AccuracyResult):
    def __init__(self):
        super(MofdictionaryAccuracyResult, self).__init__('mof')
        for data_name in ['mof', 'm_precursor', 'm_composition', 'o_precursor', 'o_composition',
                          's_precursor', 'time', 'temperature']:
            self.data[data_name] = ComputableNamedList('tp fp fn')

    def compare(self, mofdict1, mofdict2, composition_threshold=0.01, name_threshold=0.8):
        result = _compare_mofdictionary(mofdict1, mofdict2, composition_threshold, name_threshold)
        self.update(result)
        return self


def _most_similar_target(target1, list_target, threshold=0.8):
    total_acc = 0
    total_target = []
    for target2 in list_target:
        acc = fuzz.partial_ratio(target1['name'], target2['name']) / 100
        if acc < threshold:
            pass
        elif acc > total_acc:
            total_acc = acc
            total_target.clear()
            total_target.append(target2)
        elif acc == total_acc:
            total_target.append(target2)

    return total_target


def _compare_precursor(pre1, pre2, composition_threshold=0.01):
    result = PrecursorAccuracyResult()
    if isinstance(pre2, (list, MofDictionary)):
        for pre_ in pre2:
            result_ = _compare_precursor(pre1, pre_, composition_threshold)
            result.update(result_)
        return result

    elif isinstance(pre2, type_precursor):
        acc_name = fuzz.partial_ratio(pre1['name'], pre2['name']) / 100
        result.name_accuracy = acc_name

        pre1_dict = defaultdict(list)
        for value_tuple in pre1['composition']:
            value, unit = transform_unit(value_tuple, float_type='float')
            pre1_dict[unit].append(value)

        pre2_dict = defaultdict(list)
        for value_tuple in pre2['composition']:
            value, unit = transform_unit(value_tuple, float_type='float')
            pre2_dict[unit].append(value)

        for unit, values in pre1_dict.items():
            compare_values = pre2_dict[unit]
            for value in values:
                activation = False
                if value is None:
                    continue

                for c_value in compare_values:
                    try:
                        score = abs((value-c_value)/value)
                    except (ZeroDivisionError, RuntimeWarning):
                        score = c_value
                    except TypeError:
                        continue

                    if score < composition_threshold:
                        activation = True
                        result['composition'].tp += 1
                        break
                if not activation:
                    result['composition'].fp += 1

        for unit, values in pre2_dict.items():
            compare_values = pre1_dict[unit]
            for value in values:
                activation = False
                if value is None:
                    continue

                for c_value in compare_values:
                    try:
                        score = abs((value - c_value) / value)
                    except ZeroDivisionError:
                        if c_value == 0:
                            score = 0
                        else:
                            score = value
                    except TypeError:
                        continue

                    if score < composition_threshold:
                        activation = True
                        break
                if not activation:
                    result['composition'].fn += 1

        return result


def _compare_mof(mof1, mof2, composition_threshold=0.01, name_threshold=0.8):
    result = MofAccuracyResult()

    if isinstance(mof2, list):
        for pre_ in mof2:
            result_ = _compare_mof(mof1, pre_, composition_threshold)
            result.update(result_)
        return result

    elif isinstance(mof2, MOF):
        acc_name = fuzz.partial_ratio(mof1.name, mof2.name) / 100
        result.name_accuracy = acc_name

        m_result = PrecursorAccuracyResult()
        for M_pre in mof1.M_precursor:
            compare_m_pre = _most_similar_target(M_pre, mof2.M_precursor, name_threshold)
            if not compare_m_pre:
                result['m_precursor'].fp += 1
            else:
                result['m_precursor'].tp += 1
                m_result.update(_compare_precursor(M_pre, compare_m_pre))
        for M_pre in mof2.M_precursor:
            compare_m_pre = _most_similar_target(M_pre, mof1.M_precursor, name_threshold)
            if not compare_m_pre:
                result['m_precursor'].fn += 1
        result.data['m_composition'] = m_result.data['composition']

        o_result = PrecursorAccuracyResult()
        for O_pre in mof1.O_precursor:
            compare_o_pre = _most_similar_target(O_pre, mof2.O_precursor, name_threshold)
            if not compare_o_pre:
                result['o_precursor'].fp += 1
            else:
                result['o_precursor'].tp += 1
                o_result.update(_compare_precursor(O_pre, compare_o_pre))
        for O_pre in mof2.O_precursor:
            compare_o_pre = _most_similar_target(O_pre, mof1.O_precursor, name_threshold)
            if not compare_o_pre:
                result['o_precursor'].fn += 1
        result.data['o_composition'] = o_result.data['composition']

        for S_pre in mof1.S_precursor:
            compare_s_pre = _most_similar_target(S_pre, mof2.S_precursor, name_threshold)
            if not compare_s_pre:
                result['s_precursor'].fp += 1
            else:
                result['s_precursor'].tp += 1
        for S_pre in mof2.S_precursor:
            compare_s_pre = _most_similar_target(S_pre, mof1.S_precursor, name_threshold)
            if not compare_s_pre:
                result['s_precursor'].fn += 1

        time1, unit1 = transform_unit(mof1.time, float_type='float')
        time2, unit2 = transform_unit(mof2.time, float_type='float')
        if time1 is None and time2 is None:
            result['time'].tp += 1
        elif unit1 == unit2:
            try:
                score = abs((time1 - time2) / time1)
            except (ZeroDivisionError, RuntimeWarning):
                score = time2
            except TypeError:
                score = 1

            if score < composition_threshold:
                result['time'].tp += 1
            else:
                result['time'].fn += 1
                result['time'].fp += 1

        temp1, unit1 = transform_unit(mof1.temperature, float_type='float')
        temp2, unit2 = transform_unit(mof2.temperature, float_type='float')
        if temp1 is None and temp2 is None:
            result['temperature'].tp += 1
        elif unit1 == unit2:
            try:
                score = abs((temp1 - temp2) / temp1)
            except (ZeroDivisionError, RuntimeWarning):
                score = temp2
            except TypeError:
                score = 1

            if score < composition_threshold:
                result['temperature'].tp += 1
            else:
                result['temperature'].fn += 1
                result['temperature'].fp += 1

        return result
    else:
        raise TypeError()


def _compare_mofdictionary(mofdict1, mofdict2, composition_threshold=0.01, name_threshold=0.8):
    result = MofdictionaryAccuracyResult()
    for mof in mofdict1:
        compare_mofs = _most_similar_target(mof, mofdict2, name_threshold)
        if not compare_mofs:
            result['mof'].fp += 1
        else:
            result['mof'].tp += 1
            result.update(_compare_mof(mof, compare_mofs, composition_threshold, name_threshold))
    for mof in mofdict2:
        compare_mofs = _most_similar_target(mof, mofdict1, name_threshold)
        if not compare_mofs:
            result['mof'].fn += 1

    return result


def compare(object1, object2):
    if isinstance(object1, MOF) and isinstance(object2, (MOF, list)):
        return _compare_mof(object1, object2)
    elif isinstance(object1, type_precursor) and isinstance(object2, (type_precursor, list)):
        return _compare_precursor(object1, object2)
    elif isinstance(object1, MofDictionary) and isinstance(object2, (MofDictionary, list)):
        return _compare_mofdictionary(object1, object2)
    else:
        raise TypeError(f'expected MOF or Precursor, but {type(object1)} and {type(object2)}')
