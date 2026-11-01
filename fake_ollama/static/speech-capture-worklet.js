class SpeechCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.samples = new Int16Array(1600);
    this.offset = 0;
    this.port.onmessage = event => {
      if (event.data === 'flush') {
        if (this.offset) this.port.postMessage(this.samples.slice(0, this.offset).buffer);
        this.offset = 0;
        this.port.postMessage('flushed');
      }
    };
  }
  process(inputs) {
    const channels = inputs[0];
    const size = channels?.[0]?.length || 128;
    for (let i = 0; i < size; i++) {
      let value = 0;
      if (channels?.length) {
        for (const channel of channels) value += channel[i] || 0;
        value /= channels.length;
      }
      this.samples[this.offset++] = Math.round(Math.max(-1, Math.min(1, value)) * 32767);
      if (this.offset === this.samples.length) {
        this.port.postMessage(this.samples.buffer, [this.samples.buffer]);
        this.samples = new Int16Array(1600);
        this.offset = 0;
      }
    }
    return true;
  }
}
registerProcessor('speech-capture', SpeechCapture);
