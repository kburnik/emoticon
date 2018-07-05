
var sampleEmoticon = new Emoticon({
  scale: 10,
  box: {
    margin: 1,
    width: 50,
    height: 50,
  },
  head: {
    lineWidth: 1,
  },
  eyes: {
    centerDistance: 1/6,
    offset: 1/5,
    lineWidth: 1/2,
    left: {
      size: 1.5,
      border: 4,
    },
    right: {
      size: 1.5,
      border: 4
    }
  },
  eyeBrows: {
    distance: 1.5,
    offset: 1/7,
    lineWidth: 1,
    left: {
      arc: Arc.of(1/4, 3/4)
    },
    right: {
      arc: Arc.of(1/4, 3/4)
    }
  },
  mouth: {
    arc: Arc.of(0, 2),
    centerDistance: 1/5,
    offset: 1/2,
    lineWidth: 1/2,
    flip: false,
    fill: true
  }
});

var happyBase = {
  head: {
    lineWidth: 1,
  },
  eyes: {
    centerDistance: 1/8,
    offset: 1/5,
    lineWidth: 1/2,
    size: 1.2,
    border: 4
  },
  eyeBrows: {
    distance: 1.8,
    offset: 1/10,
    lineWidth: 1/2,
    arc: Arc.of(1/4, 3/4),
  },
  mouth: {
    arc: Arc.of(1/4, 3/4),
    centerDistance: 3/5,
    offset: 0,
    lineWidth: 1/2,
    flip: false,
    fill: false
  }
};

var happy = new ConfigSpace(Object.assign({}, happyBase, {

  head: {
    lineWidth: Choice.of([0.1, 0.5, 1]),
  },

  eyes: {
    centerDistance: Choice.of([1/8, 1/7, 1/5]),
    offset: Choice.of([1/10, 1/6]),
    lineWidth: 1/2,
    size: Choice.of([0.8, 1.5, 3]),
    border: Choice.of([0.1, 1, 2])
  },

  eyeBrows: {
    distance: Choice.of([1.8, 1.9]),
    offset: 1/10,
    lineWidth: Choice.of([1/1000, 1/2, 1]),
    arc: Choice.of([ Arc.of(1/4, 3/4), Arc.of(1/3, 2/3) ])
  },

  mouth: {
    arc: Choice.of([ Arc.of(1/4, 3/4), Arc.of(1/3, 2/3) ]),
    centerDistance: Choice.of([3/5, 4/5]),
    offset: 0,
    lineWidth: Choice.of([1/2, 1]),
    flip: false,
    fill: Choice.of([true, false])
  }

}));


var surprised = new ConfigSpace(Object.assign({}, happyBase, {

  head: {
    lineWidth: Choice.of([0.1, 0.5, 1]),
  },

  eyes: {
    centerDistance: Choice.of([1/8, 1/7, 1/5]),
    offset: Choice.of([1/10, 1/6]),
    lineWidth: 1/2,
    size: Choice.of([0.8, 1.5, 3]),
    border: Choice.of([0.1, 1, 2])
  },

  eyeBrows: {
    distance: 1.2,
    offset: 1/50,
    lineWidth: Choice.of([1/1000, 1/2, 1]),
    arc: Choice.of([ Arc.of(1/4, 3/4) ])
  },

  mouth: {
    arc: Arc.of(0, 2),
    centerDistance: Choice.of([0.1, 0.2]),
    offset: Choice.of([0.4, 0.6]),
    lineWidth: Choice.of([0.5, 1]),
    flip: false,
    fill: Choice.of([true, false])
  }
}));




var sadBase = {
  head: {
    lineWidth: 1,
  },
  eyes: {
    centerDistance: 1/8,
    offset: 1/5,
    lineWidth: 1/2,
    size: 1.2,
    border: 1
  },
  eyeBrows: {
    distance: 1.8,
    offset: 1/10,
    lineWidth: 1/2,
    arc: Arc.of(1/4, 3/4),
  },
  mouth: {
    arc: Arc.of(1/4, 3/4),
    centerDistance: 3/5,
    offset: 0,
    lineWidth: 1/2,
    flip: false,
    fill: false
  }
};

var sad = new ConfigSpace(Object.assign({}, sadBase, {

  head: {
    lineWidth: Choice.of([0.1, 0.5, 1]),
  },

  eyes: {
    centerDistance: Choice.of([1/8, 1/7, 1/5]),
    offset: Choice.of([1/10, 1/6]),
    lineWidth: 1/2,
    size: Choice.of([0.8, 1.5, 3]),
    border: Choice.of([0.1, 1, 2])
  },

  eyeBrows: {
    distance: Choice.of([1.8, 1.9]),
    offset: 1/10,
    lineWidth: Choice.of([1/1000, 1/2, 1]),
    arc: Choice.of([ Arc.of(1/4, 3/4), Arc.of(1/3, 2/3) ])
  },

  mouth: {
    arc: Choice.of([ Arc.of(1/4, 3/4) ]),
    centerDistance: Choice.of([3/5, 2/5]),
    offset: 0,
    lineWidth: Choice.of([1/2, 1]),
    flip: true,
    fill: false
  }

}));


