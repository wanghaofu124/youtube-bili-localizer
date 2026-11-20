// Mutations must reach the server in order; only the latest request may update the view.
export function createSiteRequests() {
  let revision = 0;
  let pending: Promise<void> = Promise.resolve();

  const latest = async <T>(request: Promise<T | null>, token: number): Promise<T | null> => {
    try {
      const result = await request;
      return token === revision ? result : null;
    } catch (error) {
      if (token !== revision) return null;
      throw error;
    }
  };

  return {
    read<T>(load: () => Promise<T>): Promise<T | null> {
      const token = ++revision;
      return latest(pending.then(() => token === revision ? load() : null), token);
    },
    write<T>(save: () => Promise<T>): Promise<T | null> {
      const token = ++revision;
      const request = pending.then(save);
      pending = request.then(() => undefined, () => undefined);
      return latest(request, token);
    },
  };
}
