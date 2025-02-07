module.exports = {
  apps: [
    {
      name: 'synth-miner-74',
      interpreter: 'python3',
      script: './neurons/miner.py',
      args: '--netuid 50 --logging.debug --logging.trace --wallet.name wgck2 --wallet.hotkey wghk54 --axon.port 18092 --blacklist.force_validator_permit true --blacklist.validator_min_stake 1000',
      env: {
        PYTHONPATH: '.'
      },
    },
  ],
};
